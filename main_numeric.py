"""
Numerical simulation.

This code builds on the following projects with adaptations:
- https://github.com/imantdaunhawer/multimodal-contrastive-learning
"""

import argparse
import json
import os
import random
import uuid
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import wishart
from sklearn import kernel_ridge, linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import clip_grad_norm_

import encoders
from utils.invertible_network_utils import construct_invertible_mlp
from utils.latent_spaces import LatentSpace, NRealSpace, ProductLatentSpace
from utils.losses import LpSimCLRLoss
from utils.infinite_iterator import PowersetIndexer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--encoding-size", type=int, default=0)         # dimensionality of encoders output, dim(\hat{z})
    parser.add_argument("--semantics-n", type=int, default=10)          # dimensionality of the latent semantic subspace, dim(s)
    parser.add_argument("--modality-n", type=int, default=5)            # dimensionality of the non-semantic modality-specific subspace, dim(m_x) and dim (m_t)
    parser.add_argument("--theta-value", type=int, default=1022)           # fixed selection bias.  ->   0, 10, 55, 175, 385, 637, 847, 967, 1012, 1022 for prefixes (e.g., 0 for indices [1], 1022 for [1,... ,10])
    parser.add_argument("--beta-value", type=int, default=-1)            # fixed perturbation bais, -1 for empty subset
    parser.add_argument("--change-prob", type=float, default=0.75)      # to reflect allowing a subet of I_beta to change at a time if prob < 1
    parser.add_argument("--causal-dependence", action="store_true")     # allowing unknown causal relations among semantics
    parser.add_argument("--margin-param", type=float, default=1.0)      # marginal concertration parameter
    parser.add_argument("--cond-param", type=float, default=1.0)        # conditional concertraintion parameter
    parser.add_argument("--n-mixing-layer", type=int, default=3)
    parser.add_argument("--seed", type=int, default=np.random.randint(32**2-1))
    parser.add_argument("--batch-size", type=int, default=6144)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-steps", type=int, default=100001)
    parser.add_argument("--log-steps", type=int, default=1000)
    parser.add_argument("--evaluate", action='store_true')              # to evaluate other than training
    parser.add_argument("--num-eval-batches", type=int, default=5)      
    parser.add_argument("--mlp-eval", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")  
    parser.add_argument("--load-args", action="store_true")
    args = parser.parse_args()

    return args, parser


def train_step(data, h_x, h_t, loss_func, optimizer, params):

    # reset grad
    if optimizer is not None:
        optimizer.zero_grad()

    # compute symmetrized loss
    z_x, z_t, z_x_, z_t_ = data
    hz_x = h_x(z_x)
    hz_x_ = h_x(z_x_)
    hz_t = h_t(z_t)
    hz_t_ = h_t(z_t_)
    total_loss_value1, _, _ = loss_func(z_x, z_t, z_x_, hz_x, hz_t, hz_x_)
    total_loss_value2, _, _ = loss_func(z_t, z_x, z_t_, hz_t, hz_x, hz_t_)
    total_loss_value = 0.5 * (total_loss_value1 + total_loss_value2)

    # backprop
    if optimizer is not None:
        total_loss_value.backward()
        clip_grad_norm_(params, max_norm=2.0, norm_type=2)  # stabilizes training
        optimizer.step()

    return total_loss_value.item()


def val_step(data, h_x, h_t, loss_func):
    return train_step(data, h_x, h_t, loss_func, optimizer=None, params=None)


def generate_data(latent_space, h_x, h_t, device, num_batches=1, batch_size=4096, loss_func=None):
    n_s = latent_space.semantics_n
    perturbed_idx = latent_space.perturb_indices
    inv_idx = latent_space.inv_indices

    semantic_dict = {f's_{i}': [] for i in range(n_s)}
    semantic_dict["s_inv"] = []
    perturbed_dict = {f'tilde_s_{i}': [] for i in perturbed_idx}
    non_semantic_dict = {"m_x": [], "m_t": []}
    reps_dict = {"hz_x": [], "hz_t": [], "loss_values": []}

    with torch.no_grad():
        for _ in range(num_batches):
            z_x, z_t, semantics, s_theta_tilde, m_x, m_t = latent_space.sample_zx_zt(batch_size, device)

            hz_x = h_x(z_x).cpu().numpy()
            hz_t = h_t(z_t).cpu().numpy()

            if loss_func is not None:
                z_x_, z_t_, *_ = latent_space.sample_zx_zt(batch_size, device)
                loss_value = val_step([z_x, z_t, z_x_, z_t_], h_x, h_t, loss_func)
                reps_dict["loss_values"].append([loss_value])

            semantics_np = semantics.cpu().numpy()
            for i in range(n_s):
                semantic_dict[f's_{i}'].append(semantics_np[:, i:i+1])
            semantic_dict['s_inv'].append(semantics_np[:, inv_idx])

            for i in perturbed_idx:
                perturbed_dict[f'tilde_s_{i}'].append(s_theta_tilde[:, i:i+1].cpu().numpy())

            non_semantic_dict["m_x"].append(m_x.cpu().numpy())
            non_semantic_dict["m_t"].append(m_t.cpu().numpy())

            reps_dict["hz_x"].append(hz_x)
            reps_dict["hz_t"].append(hz_t)

    data_dict = {"s": semantic_dict, "perturbed_s": perturbed_dict, "m": non_semantic_dict, "reps": reps_dict}

    for section_dict in data_dict.values():
        for key in section_dict:
            section_dict[key] = np.concatenate(section_dict[key], axis=0)

    return data_dict


def evaluate_prediction(model, metric, X_train, y_train, X_test, y_test):
    # handle edge cases when inputs or labels are zero-dimensional
    if any([0 in x.shape for x in [X_train, y_train, X_test, y_test]]):
        return np.nan
    assert X_train.shape[1] == X_test.shape[1]
    assert y_train.shape[1] == y_test.shape[1]
    # handle edge cases when the inputs are one-dimensional
    if X_train.shape[1] == 1:
        X_train = X_train.reshape(-1, 1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return metric(y_test, y_pred)


def main():
    args, _ = parse_args()
    
    if args.model_id is None:
        setattr(args, "model_id", str(uuid.uuid4()))
    args.save_dir = os.path.join(args.model_dir, args.model_id)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    if args.evaluate and args.load_args:
        with open(os.path.join(args.save_dir, "args.json"), "r") as fp:
            loaded_args = json.load(fp)
        arguments_to_load = ["change_prob", "causal_dependence", "encoding_size", "semantics_n", "modality_n",
                             "theta_value", "beta_value", "cond_param", "margin_param", "n_mixing_layer"
        ]
        for arg in arguments_to_load:
            setattr(args, arg, loaded_args[arg])
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # load training seed, which ensures consistent latent spaces for evaluation
    if args.evaluate:
        with open(os.path.join(args.save_dir, "args.json"), "r") as fp:
            train_seed = json.load(fp)["seed"]
        assert args.seed != train_seed
    else:
        train_seed = args.seed
    
    # set device
    if torch.cuda.is_available() and not args.no_cuda:
        device = "cuda"
    else:
        device = "cpu"
        warnings.warn("cuda is not available or --no-cuda was set.")
        
    # define loss function
    loss_func = LpSimCLRLoss()
    
    n_semantic, n_modality = args.semantics_n, args.modality_n    
    pset = PowersetIndexer(n_semantic)
    selected_indices = pset[args.theta_value]      # selected semantics indices for generating text caption
    
    if args.beta_value == -1:
        perturbed_indices = []
    else:
        perturbed_indices = pset[args.beta_value]
    
    # define latent space
    latent_space_list = []
    Sigma_s, Sigma_mx, Sigma_a, Sigma_mt = [None] * 4
    rgen = torch.Generator(device=device)
    rgen.manual_seed(train_seed)        # ensure same latents for train and eval
    
    # space of all semantics
    space_semantics = NRealSpace(n_semantic, selected_indices, perturbed_indices)
    if args.causal_dependence:
        Sigma_s = wishart.rvs(n_semantic, np.eye(n_semantic), size=1, random_state=train_seed)
        Sigma_a = wishart.rvs(n_semantic, np.eye(n_semantic), size=1, random_state=train_seed)
    sample_marginal_semantics = lambda space, size, device=device: \
        space.normal(None, args.margin_param, size, device, Sigma=Sigma_s)
    sample_conditional_semantics = lambda space, z, size, device=device:\
        space.normal(z, args.cond_param, size, device, change_prob=args.change_prob, Sigma=Sigma_a)
    latent_space_list.append(LatentSpace(
        space=space_semantics,
        sample_marginal=sample_marginal_semantics,
        sample_conditional=sample_conditional_semantics
    ))
    
    # modality specific spaces
    if n_modality > 0:
        if args.causal_dependence:
            Sigma_mx = wishart.rvs(n_modality, np.eye(n_modality), size=1, random_state=train_seed)
            Sigma_mt = wishart.rvs(n_modality, np.eye(n_modality), size=1, random_state=train_seed)
        
        space_mx = NRealSpace(n_modality)
        sample_marginal_mx = lambda space, size, device=device: \
            space.normal(None, args.margin_param, size, device, Sigma=Sigma_mx)
        sample_conditional_mx = lambda space, z, size, device=device: z
        latent_space_list.append(LatentSpace(
            space=space_mx,
            sample_marginal=sample_marginal_mx,
            sample_conditional=sample_conditional_mx
        ))
        
        space_mt = NRealSpace(n_modality)
        sample_marginal_mt = lambda space, size, device=device: \
            space.normal(None, args.margin_param, size, device, Sigma=Sigma_mt)
        sample_conditional_mt = lambda space, z, size, device=device: z
        latent_space_list.append(LatentSpace(
            space=space_mt,
            sample_marginal=sample_marginal_mt,
            sample_conditional=sample_conditional_mt
        ))        
    
    # combine latents
    latent_space = ProductLatentSpace(spaces=latent_space_list)
    dim_zx, dim_zt, dim_rep = latent_space.dim
    
    args.encoding_size = dim_rep if args.encoding_size == 0 else args.encoding_size
    
    # print arguments
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")
    if not args.evaluate:
        with open(os.path.join(args.save_dir, "args.json"), "w") as fp:
            json.dump(args.__dict__, fp)
    
    # define mixing functions
    g_x = construct_invertible_mlp(
        n = dim_zx,
        n_layers=args.n_mixing_layer,
        cond_thresh_ratio=0.001,
        n_iter_cond_thresh=25000
    ).to(device)
    g_t = construct_invertible_mlp(     # for the selected text view
        n=dim_zt,
        n_layers=args.n_mixing_layer,
        cond_thresh_ratio=0.001,
        n_iter_cond_thresh=25000
    ).to(device)
    
    if args.evaluate:
        g_x_path = os.path.join(args.save_dir, 'g_x.pt')
        g_x.load_state_dict(torch.load(g_x_path, map_location=device))
        g_t_path = os.path.join(args.save_dir, 'g_t.pt')
        g_t.load_state_dict(torch.load(g_t_path, map_location=device))
    
    # freeze
    for p in g_x.parameters():
        p.requires_grad = False
    for p in g_t.parameters():
        p.requires_grad = False
    # save mixing functions tor disk
    if args.save_dir and not args.evaluate:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(g_x.state_dict(), os.path.join(args.save_dir, "g_x.pt"))
        torch.save(g_t.state_dict(), os.path.join(args.save_dir, "g_t.pt"))
    
    # define encoders
    f_x = encoders.get_mlp(
        n_in=dim_zx,
        n_out=args.encoding_size,
        layers=[dim_zx * 10,
                dim_zx * 50,
                dim_zx * 50,
                dim_zx * 50,
                dim_zx * 50,
                dim_zx * 10]
    ).to(device)
    f_t = encoders.get_mlp(
        n_in=dim_zt,
        n_out=args.encoding_size,
        layers=[dim_zt * 10,
                dim_zt * 50,
                dim_zt * 50,
                dim_zt * 50,
                dim_zt * 50,
                dim_zt * 10]
    ).to(device)
    
    if args.evaluate:
        f_x_path = os.path.join(args.save_dir, 'f_x.pt')
        f_x.load_state_dict(torch.load(f_x_path, map_location=device))
        f_t_path = os.path.join(args.save_dir, 'f_t.pt')
        f_t.load_state_dict(torch.load(f_t_path, map_location=device))
        
    h_x = lambda z: f_x(g_x(z))
    h_t = lambda z: f_t(g_t(z))
    
    if not args.evaluate:
        params = list(f_x.parameters()) + list(f_t.parameters())
        optimizer = torch.optim.Adam(params, lr=args.lr)
    
    # training
    # ----------
    step = 1
    while step <= args.train_steps and not args.evaluate:
        
        # training step in distribution
        z_x, z_t, *_ = latent_space.sample_zx_zt(args.batch_size, device)
        z_x_, z_t_, *_ = latent_space.sample_zx_zt(args.batch_size, device)
        data = [z_x, z_t, z_x_, z_t_]
            
        train_step(data, h_x, h_t, loss_func, optimizer, params)

        if step % args.log_steps == 1 or step == args.train_steps:
            
            # save encoders to disk
            if args.save_dir and not args.evaluate:
                torch.save(f_x.state_dict(), os.path.join(args.save_dir, "f_x.pt"))
                torch.save(f_t.state_dict(), os.path.join(args.save_dir, "f_t.pt"))
            
            # lightweight evaluation with linear classifiers
            print(f"\nStep: {step} \t")
            data_dict = generate_data(latent_space, h_x, h_t, device, loss_func=loss_func)
            print(f"<Loss>: {np.mean(data_dict['reps']['loss_values']):.4f} \t")
            data_dict['reps']['hz_x'] = StandardScaler().fit_transform(data_dict['reps']['hz_x'])
            for sec in ['s', 'perturbed_s', 'm']:
                keys = data_dict[sec].keys()
                for k in keys:
                    inputs, labels = data_dict['reps']['hz_x'], data_dict[sec][k]
                    train_inputs, test_inputs, train_labels, test_labels = \
                        train_test_split(inputs, labels)
                    data = [train_inputs, train_labels, test_inputs, test_labels]
                    r2_linear = evaluate_prediction(
                        linear_model.LinearRegression(n_jobs=-1), r2_score, *data)
                    print(f"{k} r2_linear: {r2_linear}")
        step += 1
    
    
    # evaluation
    # ----------
    # if args.evaluate:     # evaluate immediately after training
    
    # generate encoding and labels for the validation and test data
    val_dict = generate_data(
        latent_space, h_x, h_t, device, 
        num_batches=args.num_eval_batches,
        loss_func=loss_func
    )
    test_dict = generate_data(
        latent_space, h_x, h_t, device,
        num_batches=args.num_eval_batches,
        loss_func=loss_func
    )
    
    # print average loss value
    print(f"<Val Loss>: {np.mean(val_dict['reps']['loss_values']):.4f} \t")
    print(f"<Test Loss>: {np.mean(test_dict['reps']['loss_values']):.4f} \t")

    # standardize the encodings
    for m in ['x', 't']:
        scaler = StandardScaler()
        val_dict['reps'][f"hz_{m}"] = scaler.fit_transform(val_dict['reps'][f"hz_{m}"])
        test_dict['reps'][f"hz_{m}"] = scaler.transform(test_dict['reps'][f"hz_{m}"])
    
    # train predictors on data from val_dict and evaluate on test_dict
    results = []
    for m in ['x', 't']:
        for sec in ['s', 'perturbed_s', 'm']:
            section_dict = val_dict[sec]
            for k in section_dict:
                
                # select data
                train_inputs, test_inputs = val_dict['reps'][f"hz_{m}"], test_dict['reps'][f"hz_{m}"]
                train_labels, test_labels = val_dict[sec][k], test_dict[sec][k]
                data = [train_inputs, train_labels, test_inputs, test_labels]
                
                # linear regression
                r2_linear = evaluate_prediction(
                    linear_model.LinearRegression(n_jobs=-1), r2_score, *data
                )
                
                # nonlinear regression
                if args.mlp_eval:
                    model = MLPRegressor(max_iter=10000)  # lightweight option
                else:
                    # grid search is time- and memory-intensive
                    model = GridSearchCV(
                        kernel_ridge.KernelRidge(kernel='rbf', gamma=0.1),
                        param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                    "gamma": np.logspace(-2, 2, 4)},
                        cv=3, n_jobs=-1)
                r2_nonlinear = evaluate_prediction(model, r2_score, *data)
                
                # append results
                results.append((f"hz_{m}", k, r2_linear, r2_nonlinear))
    
    # convert evaluation results into tabular form
    cols = ["encoding", "predicted_factors", "r2_linear", "r2_nonlinear"]
    df_results = pd.DataFrame(results, columns=cols)
    df_results.to_csv(os.path.join(args.save_dir, "results.csv"))
    print("Regression results:")
    print(df_results.to_string())

if __name__ == "__main__":
    main()