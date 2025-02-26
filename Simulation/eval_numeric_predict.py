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
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import clip_grad_norm_

import encoders
from utils.invertible_network_utils import construct_invertible_mlp
from utils.latent_spaces import LatentSpace, NRealSpace, ProductLatentSpace
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
    parser.add_argument("--num-eval-batches", type=int, default=5)  
    parser.add_argument("--mlp-eval", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")  
    args = parser.parse_args()

    return args, parser



def polynomial_nonlinear_function(z: torch.Tensor) -> torch.Tensor:
    """
    A non-linear polynomial function mapping an input tensor z (of any dimension)
    to a scalar value for regression labeling.
    
    The function includes:
    - Quadratic terms
    - Interaction terms (pairwise products)
    - Sinusoidal transformation
    
    Parameters:
    - z: torch.Tensor of shape (n_samples, n_features)
    
    Returns:
    - f_z: torch.Tensor of shape (n_samples,), the computed regression values
    """
    dim = z.shape[1]
    
    # Quadratic terms
    quad_terms = torch.sum(z**2, dim=1)
    
    # Interaction terms (pairwise products)
    interaction_terms = torch.sum(torch.stack([z[:, i] * z[:, j] for i in range(dim) for j in range(i+1, dim)]), dim=0)
    
    # Sinusoidal transformation of the first dimension
    sinusoidal_component = torch.sin(z[:, 0])
    
    # Final function combination
    f_z = quad_terms + 0.5 * interaction_terms + sinusoidal_component
    
    return f_z


def generate_data(latent_space, h_x, h_t, device, num_batches=1, batch_size=4096):
    target_lables = {'y1':[], 'y2':[], 'y3':[]}
    reps_dict = {"hz_x":[], "hz_t":[]}
    
    with torch.no_grad():
        for _ in range(num_batches):
            
            # sample batch of latents
            z_x, z_t, semantics, s_theta_tilde, m_x, m_t = latent_space.sample_zx_zt(batch_size, device)
            
            # compute representations
            hz_x = h_x(z_x)
            hz_t = h_t(z_t)
            
            # collect labels and representations
            y1 = polynomial_nonlinear_function(semantics[:, 0:3])   # [s1, s2, s3] -> y1
            y2 = polynomial_nonlinear_function(semantics[:, 0:5])   # [s1, s2, s3, s4, s5] -> y2
            y3 = polynomial_nonlinear_function(semantics[:, 0:7])   # [s1, s2, s3, s4, s5, s6, s7] -> y3s
            target_lables["y1"].append(y1.unsqueeze(-1).detach().cpu().numpy())
            target_lables["y2"].append(y2.unsqueeze(-1).detach().cpu().numpy())
            target_lables["y3"].append(y3.unsqueeze(-1).detach().cpu().numpy())
            
            reps_dict["hz_x"].append(hz_x.detach().cpu().numpy())
            reps_dict["hz_t"].append(hz_t.detach().cpu().numpy())
    
    data_dict = {"labels":target_lables, "reps":reps_dict}
    
    for section in data_dict:
        for k, v in data_dict[section].items():
            if len(v) > 0:
                v = np.concatenate(v, axis=0)
            data_dict[section][k] = np.array(v)
    
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
    with open(os.path.join(args.save_dir, "args.json"), "r") as fp:
        train_seed = json.load(fp)["seed"]
    assert args.seed != train_seed
    
    # set device
    if torch.cuda.is_available() and not args.no_cuda:
        device = "cuda"
    else:
        device = "cpu"
        warnings.warn("cuda is not available or --no-cuda was set.")

    n_semantic, n_modality = args.semantics_n, args.modality_n    
    pset = PowersetIndexer(n_semantic)
    selected_indices = pset[args.theta_value]      # selected semantics indices for generating text caption
    
    if args.beta_value == -1:
        perturbed_indices = []
    else:
        perturbed_indices = pset[args.beta_value]
    
    # define latent space
    latent_space_list = []
    shifted_space_list = []     # distribution shift latext space
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
    
    # shifted semantics space
    # distribution shift occurs in [s_6, s_7, s_8, s_9, s_10] -> mean shift
    space_shift_semantics = NRealSpace(n_semantic, selected_indices, perturbed_indices)
    sample_marginal_shifted = lambda space, size, device=device: \
        space.normal(torch.Tensor([0]*(n_semantic//2) + [0.1]*(n_semantic-n_semantic//2))
                     , args.margin_param, size, device, Sigma=Sigma_s)
    shifted_space_list.append(LatentSpace(
        space=space_shift_semantics,
        sample_marginal=sample_marginal_shifted,
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
        shifted_space_list.append(LatentSpace(
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
        shifted_space_list.append(LatentSpace(
            space=space_mt,
            sample_marginal=sample_marginal_mt,
            sample_conditional=sample_conditional_mt
        ))       
    
    # combine latents
    latent_space = ProductLatentSpace(spaces=latent_space_list)
    shifted_space = ProductLatentSpace(spaces=shifted_space_list)
    dim_zx, dim_zt, dim_rep = latent_space.dim
    
    args.encoding_size = dim_rep if args.encoding_size == 0 else args.encoding_size
    
    # print arguments
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

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
    
    g_x_path = os.path.join(args.save_dir, 'g_x.pt')
    g_x.load_state_dict(torch.load(g_x_path, map_location=device))
    g_t_path = os.path.join(args.save_dir, 'g_t.pt')
    g_t.load_state_dict(torch.load(g_t_path, map_location=device))
    
    # freeze
    for p in g_x.parameters():
        p.requires_grad = False
    for p in g_t.parameters():
        p.requires_grad = False

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
    
    f_x_path = os.path.join(args.save_dir, 'f_x.pt')
    f_x.load_state_dict(torch.load(f_x_path, map_location=device))
    f_t_path = os.path.join(args.save_dir, 'f_t.pt')
    f_t.load_state_dict(torch.load(f_t_path, map_location=device))

    h_x = lambda z: f_x(g_x(z))
    h_t = lambda z: f_t(g_t(z))
    
    # generate encoding and labels for the validation and test data
    val_dict = generate_data(
        latent_space, h_x, h_t, device, 
        num_batches=args.num_eval_batches
    )
    test_iid_dict = generate_data(
        latent_space, h_x, h_t, device,
        num_batches=args.num_eval_batches
    )
    test_ood_dict = generate_data(
        shifted_space, h_x, h_t, device,
        num_batches=args.num_eval_batches
    )
    

    # standardize the encodings
    for m in ['x', 't']:
        scaler = StandardScaler()
        val_dict['reps'][f"hz_{m}"] = scaler.fit_transform(val_dict['reps'][f"hz_{m}"])
        test_iid_dict['reps'][f"hz_{m}"] = scaler.transform(test_iid_dict['reps'][f"hz_{m}"])
        test_ood_dict['reps'][f"hz_{m}"] = scaler.transform(test_ood_dict['reps'][f"hz_{m}"])
    
    # train predictors on data from val_dict and evaluate on test_dict
    results = []
    for m in ['x', 't']:
        for sec in ['labels']:
            section_dict = val_dict[sec]
            for k in section_dict:
                
                # select data
                train_inputs, test_iid_inputs, test_ood_inputs = val_dict['reps'][f"hz_{m}"], test_iid_dict['reps'][f"hz_{m}"], test_ood_dict['reps'][f"hz_{m}"]
                train_labels, test_iid_labels, test_ood_labels = val_dict[sec][k], test_iid_dict[sec][k], test_ood_dict[sec][k]
                data_iid = [train_inputs, train_labels, test_iid_inputs, test_iid_labels]
                data_ood = [train_inputs, train_labels, test_ood_inputs, test_ood_labels]
                
                
                # nonlinear regression
                model = MLPRegressor(max_iter=10000)  # lightweight option
                r2_nonlinear_iid = evaluate_prediction(model, r2_score, *data_iid)
                r2_nonlinear_ood = evaluate_prediction(model, r2_score, *data_ood)
                
                # append results
                results.append((f"hz_{m}", k, r2_nonlinear_iid, r2_nonlinear_ood))
    
    # convert evaluation results into tabular form
    cols = ["encoding", "predicted_factors", "r2_nonlinear_iid", "r2_nonlinear_ood"]
    df_results = pd.DataFrame(results, columns=cols)
    df_results.to_csv(os.path.join(args.save_dir, "results_predict.csv"))
    print("Downstream Predict results:")
    print(df_results.to_string())


if __name__ == "__main__":
    main()