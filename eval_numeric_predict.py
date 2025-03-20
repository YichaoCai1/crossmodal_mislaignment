import argparse
import json
import os
import random
import uuid
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import wishart
from sklearn.metrics import r2_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import encoders
from utils.invertible_network_utils import construct_invertible_mlp
from utils.latent_spaces import LatentSpace, NRealSpace, ProductLatentSpace
from utils.infinite_iterator import PowersetIndexer


def parse_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--no-cuda", action="store_true")  
    args = parser.parse_args()

    return args, parser


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 100),  # Hidden layer (100 neurons)
            nn.ReLU(),                  # Activation function
            nn.Linear(100, output_dim)  # Output layer (Identity activation)
        )
        self._initialize_weights()  # Apply Xavier initialization

    def _initialize_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Equivalent to Glorot initialization
                nn.init.zeros_(m.bias)  # Bias set to zero

    def forward(self, x):
        return self.model(x)  # Linear output for regression


def train_mlp_regressor(model, criterion, optimizer, X_train, y_train, device, epochs=100):
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(device),\
        torch.tensor(y_train, dtype=torch.float32).to(device)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
    return model

def train_mlp_classifier(model, optimizer, X_train, y_train, device, epochs=100):
    # Convert to PyTorch tensors and move to device
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)  # Ensure integer class labels
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train)  # Output shape: (batch_size, num_classes)
        
        # Compute loss
        loss = criterion(outputs, y_train.squeeze(dim=1))  # CrossEntropyLoss expects logits & integer labels
        
        # Backpropagation
        loss.backward()
        optimizer.step()

    return model

def complex_nonlinear_function(z: torch.Tensor) -> torch.Tensor:
    dim = z.shape[1]
    
    # Quadratic terms
    quad_terms = torch.sum(z**2, dim=1)
    
    # Cubic terms
    cubic_terms = torch.sum(z**3, dim=1)
    
    # Interaction terms (pairwise products)
    interaction_terms = torch.sum(torch.stack([z[:, i] * z[:, j] for i in range(dim) for j in range(i+1, dim)]), dim=0)
    
    # Higher-order interactions (triple-wise products)
    triple_interactions = torch.sum(torch.stack([
        z[:, i] * z[:, j] * z[:, k] for i in range(dim) for j in range(i+1, dim) for k in range(j+1, dim)
    ]), dim=0)
    
    # Sinusoidal transformation
    sinusoidal_terms = torch.sum(torch.sin(z), dim=1) + torch.sum(torch.cos(z), dim=1)
    
    # Logarithmic transformation
    log_terms = torch.sum(torch.log1p(torch.abs(z)), dim=1)  # log(1 + |z|)
    
    # Exponential transformation
    exp_terms = torch.sum(torch.exp(-torch.abs(z)), dim=1)  # e^(-|z|)
    
    # Final function combination
    f_z = (
        quad_terms + 0.3 * cubic_terms + 0.5 * interaction_terms + 
        0.2 * triple_interactions + 0.7 * sinusoidal_terms + 
        0.4 * log_terms + 0.6 * exp_terms
    )
    
    return f_z


# def gen_class_labels(z: torch.Tensor, device, num_classes: int = 10) -> torch.Tensor:
#     """
#     Generates classification labels based on input features using quantile-based binning.
#     """
#     f_z = complex_nonlinear_function(z).to(device)

#     # Compute bins dynamically from data distribution
#     bins = torch.quantile(f_z, torch.linspace(0, 1, num_classes + 1, device=device))

#     # Assign labels based on bins
#     labels = torch.bucketize(f_z, bins) - 1  # Ensure range [0, num_classes-1]
    
#     return torch.clamp(labels, 0, num_classes - 1)


def gen_class_labels(z: torch.Tensor, device, num_classes: int = 10) -> torch.Tensor:
    """
    Generates classification labels using K-Means clustering.
    
    Args:
        z (torch.Tensor): Input feature tensor of shape (N, D), where N is the number of samples.
        device (torch.device): Device for tensor computations.
        num_classes (int): Number of clusters (classes).
    
    Returns:
        torch.Tensor: Cluster-based class labels of shape (N,)
    """
    z_np = z.cpu().numpy()  # Convert to NumPy for clustering

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_classes, n_init=10, random_state=42)
    cluster_assignments = kmeans.fit_predict(z_np)  # Get cluster labels

    # Map cluster indices to class labels based on cluster centroids order
    cluster_centers = kmeans.cluster_centers_
    sorted_indices = torch.argsort(torch.tensor(cluster_centers[:, 0]))  # Sort by first dimension
    cluster_to_label = {int(cluster): int(label) for label, cluster in enumerate(sorted_indices)}

    # Convert cluster assignments to class labels
    labels = torch.tensor([cluster_to_label[c] for c in cluster_assignments], device=device)

    return labels


def generate_data(latent_space, h_x, device, num_batches=1, batch_size=4096):
    target_lables = {'y1':[], 'y2':[], 'y3':[], 'y4':[], 'y5':[]}
    reps = []
    
    with torch.no_grad():
        for _ in range(num_batches):
            
            # sample batch of latents
            z_x, _, semantics, *_ = latent_space.sample_zx_zt(batch_size, device)
            
            hz_x = h_x(z_x)
            
            # collect labels and representations
            y1 = complex_nonlinear_function(semantics[:, 0:3])   
            y2 = complex_nonlinear_function(semantics[:, 0:5])   
            y3 = complex_nonlinear_function(semantics[:, 0:7])   
            y4 = complex_nonlinear_function(semantics[:, 0:9])   
            target_lables["y1"].append(y1.unsqueeze(-1).detach().cpu().numpy())
            target_lables["y2"].append(y2.unsqueeze(-1).detach().cpu().numpy())
            target_lables["y3"].append(y3.unsqueeze(-1).detach().cpu().numpy())
            target_lables["y4"].append(y4.unsqueeze(-1).detach().cpu().numpy())
            
            y5 = gen_class_labels(y2.unsqueeze(-1), device, num_classes=2) #semantics[:, 0:5], device, num_classes=2)
            target_lables["y5"].append(y5.unsqueeze(-1).detach().cpu().numpy())
            
            reps.append(hz_x.detach().cpu().numpy())
    
    data_dict = {"labels":target_lables, "reps":np.array(np.concatenate(reps, axis=0))}
    
    for k, v in data_dict["labels"].items():
        if len(v) > 0:
            v = np.concatenate(v, axis=0)
        
        data_dict["labels"][k] = np.array(v)

    return data_dict


def main():
    args, _ = parse_args()
    
    if args.model_id is None:
        setattr(args, "model_id", str(uuid.uuid4()))
    args.save_dir = os.path.join(args.model_id)
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
    space_shift_semantics = NRealSpace(n_semantic, selected_indices, perturbed_indices)
    sample_marginal_shifted = lambda space, size, device=device: \
        space.normal(None, args.margin_param, size, device, Sigma=Sigma_s, shift_ids=[8, 9])
    sample_conditional_shifted = lambda space, z, size, device=device:\
        space.normal(z, args.cond_param, size, device, change_prob=args.change_prob, Sigma=Sigma_a)
    shifted_space_list.append(LatentSpace(
        space=space_shift_semantics,
        sample_marginal=sample_marginal_shifted,
        sample_conditional=sample_conditional_shifted
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
    
    # generate encoding and labels for the validation and test data
    val_dict = generate_data(
        latent_space, h_x, device, 
        num_batches=args.num_eval_batches
    )
    test_dict = generate_data(
        latent_space, h_x, device,
        num_batches=args.num_eval_batches
    )
    shifted_dict = generate_data(
        shifted_space, h_x, device,
        num_batches=args.num_eval_batches
    )
    
    # standardize the encodings
    scaler = StandardScaler()
    val_dict['reps'] = scaler.fit_transform(val_dict['reps'])
    test_dict['reps'] = scaler.transform(test_dict['reps'])
    shifted_dict['reps'] = scaler.transform(shifted_dict['reps'])
    
    # train predictors on data from val_dict and evaluate on test_dict
    results = []
    for k in val_dict['labels']:
        
        # select data
        train_inputs, test_inputs = val_dict['reps'], test_dict['reps']
        train_labels, test_labels = val_dict['labels'][k], test_dict['labels'][k]
                
        if k != 'y5':
            # criterion = nn.MSELoss()
            # model = MLP(input_dim=train_inputs.shape[1], output_dim=1).to(device)
            # optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # model = train_mlp_regressor(model, criterion, optimizer, train_inputs, train_labels, device, epochs=10000)
            
            # # eval
            # model.eval()
            # with torch.no_grad():
            #     y_pred = model(torch.tensor(test_inputs, dtype=torch.float32).to(device)).cpu().numpy()
            #     metric_regression = r2_score(test_labels, y_pred)
            
            # # append results
            # results.append((k, metric_regression))
            
            continue
            
        else:
            shifted_intputs, shifted_labels = shifted_dict['reps'], shifted_dict['labels'][k]
            
            model = MLP(input_dim=train_inputs.shape[1], output_dim=2).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            model = train_mlp_classifier(model, optimizer, train_inputs, train_labels, device, epochs=10000)
            
            model.eval()
            with torch.no_grad():
                pred_iid = torch.argmax(model(torch.tensor(test_inputs, dtype=torch.float32).to(device)), dim=1)
                pred_shifted = torch.argmax(model(torch.tensor(shifted_intputs, dtype=torch.float32).to(device)), dim=1)
                mcc_iid = matthews_corrcoef(test_labels, pred_iid.cpu().numpy())
                mcc_shifted = matthews_corrcoef(shifted_labels, pred_shifted.cpu().numpy())
                
                df_cls = pd.DataFrame([(mcc_iid, mcc_shifted)], columns=["iid", "ood"])
                df_cls.to_csv(os.path.join(args.save_dir, "results_classify.csv"))
                print("Classification results:")
                print(df_cls.to_string())
    
    # # convert evaluation results into tabular form
    # cols = ["task", "metric_scores"]
    # df_results = pd.DataFrame(results, columns=cols)
    # df_results.to_csv(os.path.join(args.save_dir, "results_predict.csv"))
    # print("Downstream Predict results:")
    # print(df_results.to_string())


if __name__ == "__main__":
    main()