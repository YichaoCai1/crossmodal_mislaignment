"""
Experiment with image/text pairs.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet18

from datasets import MultimodalMPI3DRealComplex
from encoders import FlexibleTextEncoder2D
from utils.infinite_iterator import InfiniteIterator
from utils.losses import infonce_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--bias-type", type=str, default="selection")
    parser.add_argument("--bias-id", type=int, default=4)
    parser.add_argument("--encoding-size", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=100)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=200001)
    parser.add_argument("--log-steps", type=int, default=1000)
    parser.add_argument("--checkpoint-steps", type=int, default=10000)
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--seed", type=int, default=np.random.randint(32**2-1))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--save-all-checkpoints", action="store_true")
    parser.add_argument("--load-args", action="store_true")
    args = parser.parse_args()
    return args, parser


def train_step(data, f1, f2, loss_func, optimizer, params):

    # reset grad
    if optimizer is not None:
        optimizer.zero_grad()

    # compute loss
    x1 = data['image']
    x2 = data['text']
    hz1 = f1(x1)
    hz2 = f2(x2)
    loss_value1 = loss_func(hz1, hz2)
    loss_value2 = loss_func(hz2, hz1)
    loss_value = 0.5 * (loss_value1 + loss_value2)  # symmetrized infonce loss

    # backprop
    if optimizer is not None:
        loss_value.backward()
        clip_grad_norm_(params, max_norm=2.0, norm_type=2)  # stabilizes training
        optimizer.step()

    return loss_value.item()


def val_step(data, f1, f2, loss_func):
    return train_step(data, f1, f2, loss_func, optimizer=None, params=None)


def get_data(dataset, f1, f2, loss_func, dataloader_kwargs):
    loader = DataLoader(dataset, **dataloader_kwargs)
    iterator = InfiniteIterator(loader)
    labels_image = {v: [] for v in MultimodalMPI3DRealComplex.VAL_LATENTS}
    rdict = {"hz_image": [], "hz_text": [], "loss_values": [],
             "semantics": labels_image}
    i = 0
    with torch.no_grad():
        while (i < len(dataset)):  # NOTE: can yield slightly too many samples

            # load batch
            i += loader.batch_size
            data = next(iterator)  # contains images, texts, and labels

            # compute loss
            loss_value = val_step(data, f1, f2, loss_func)
            rdict["loss_values"].append([loss_value])

            # collect representations
            hz_image = f1(data["image"])
            hz_text = f2(data["text"])
            rdict["hz_image"].append(hz_image.detach().cpu().numpy())
            rdict["hz_text"].append(hz_text.detach().cpu().numpy())

            # collect semantic labels
            for k in rdict["semantics"]:
                labels_k = data["semantics"][k]
                rdict["semantics"][k].append(labels_k)

    # concatenate each list of values along the batch dimension
    for k, v in rdict.items():
        if type(v) == list:
            rdict[k] = np.concatenate(v, axis=0)
        elif type(v) == dict:
            for k2, v2 in v.items():
                rdict[k][k2] = np.concatenate(v2, axis=0)
    return rdict


def evaluate_prediction(model, metric, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return metric(y_test, y_pred)


def main():

    # parse args
    args, parser = parse_args()

    # create save_dir, where the model/results are or will be saved
    if args.model_id is None:
        setattr(args, "model_id", uuid.uuid4())
    args.save_dir = os.path.join(args.model_dir, args.model_id)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # optionally, reuse existing arguments from args.json (only for evaluation)
    if args.evaluate and args.load_args:
        with open(os.path.join(args.save_dir, 'args.json'), 'r') as fp:
            loaded_args = json.load(fp)
        arguments_to_load = ["encoding_size", "hidden_size"]
        for arg in arguments_to_load:
            setattr(args, arg, loaded_args[arg])
        # NOTE: Any new arguments that shall be automatically loaded for the
        # evaluation of a trained model must be added to 'arguments_to_load'.

    # save args to disk (only for training)
    if not args.evaluate:
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as fp:
            json.dump(args.__dict__, fp)

    # set all seeds
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # set device
    if torch.cuda.is_available() and not args.no_cuda:
        device = "cuda"
    else:
        device = "cpu"
        warnings.warn("cuda is not available or --no-cuda was set.")

    # define similarity metric and loss function
    sim_metric = torch.nn.CosineSimilarity(dim=-1)
    criterion = torch.nn.CrossEntropyLoss()
    loss_func = lambda z1, z2: infonce_loss(
        z1, z2, sim_metric=sim_metric, criterion=criterion, tau=args.tau)

    # define augmentations (only normalization of the input images)
    mean_per_channel = [0.00888889, 0.00888889, 0.00830382]  # values from MPI3D-Real-Complex
    std_per_channel = [0.08381344, 0.07622504, 0.06356431]   # values from MPI3D-Real-Complex
    transform = transforms.Compose([
        transforms.Normalize(mean_per_channel, std_per_channel)])

    # define kwargs
    dataset_kwargs = {"transform": transform}
    dataloader_kwargs = {
        "batch_size": args.batch_size, "shuffle": True, "drop_last": True,
        "num_workers": args.workers, "pin_memory": True}

    # define dataloaders
    train_dataset = MultimodalMPI3DRealComplex(args.datapath, bias_type=args.bias_type, bias_id=args.bias_id, mode="train", **dataset_kwargs)
    # vocab_filepath = train_dataset.vocab_filepath
    train_loader = DataLoader(train_dataset, **dataloader_kwargs)
    train_iterator = InfiniteIterator(train_loader)
    
    test_dataset = MultimodalMPI3DRealComplex(args.datapath, bias_type=args.bias_type, bias_id=args.bias_id, mode="test",
                                        vocab_filepath=None,
                                        **dataset_kwargs)
    
    if args.encoding_size == 0:
        args.encoding_size = len(train_dataset.unbiased_semantics)
        assert args.encoding_size >= 1
    
    # print args
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")
    
    total_size = len(test_dataset)
    val_size = total_size // 2

    # Explicitly select indices
    val_indices = list(range(0, val_size))
    test_indices = list(range(val_size, total_size))

    val_dataset = Subset(test_dataset, val_indices)
    test_dataset = Subset(test_dataset, test_indices)

    # define image encoder
    encoder_img = torch.nn.Sequential(
        resnet18(num_classes=args.hidden_size),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(args.hidden_size, args.encoding_size))
    encoder_img = torch.nn.DataParallel(encoder_img)
    encoder_img.to(device)

    # define text encoder
    sequence_length = train_dataset.max_sequence_length
    print("max sequence length:", sequence_length)
    
    encoder_txt = FlexibleTextEncoder2D(
        input_size=train_dataset.vocab_size,
        output_size=args.encoding_size,
        sequence_length=sequence_length)
    encoder_txt = torch.nn.DataParallel(encoder_txt)
    encoder_txt.to(device)

    # for evaluation, always load saved encoders
    if args.evaluate:
        path_img = os.path.join(args.save_dir, "encoder_img.pt")
        path_txt = os.path.join(args.save_dir, "encoder_txt.pt")
        encoder_img.load_state_dict(torch.load(path_img, map_location=device))
        encoder_txt.load_state_dict(torch.load(path_txt, map_location=device))

    # define the optimizer
    params = list(encoder_img.parameters())+list(encoder_txt.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # training
    # --------
    if not args.evaluate:

        # training loop
        step = 1
        loss_values = []  # list to keep track of loss values
        while (step <= args.train_steps):

            # training step
            data = next(train_iterator)  # contains images, texts, and labels
            loss_value = train_step(data, encoder_img, encoder_txt, loss_func, optimizer, params)
            loss_values.append(loss_value)

            # print average loss value
            if step % args.log_steps == 1 or step == args.train_steps:
                print(f"Step: {step} \t",
                      f"Loss: {loss_value:.4f} \t",
                      f"<Loss>: {np.mean(loss_values[-args.log_steps:]):.4f} \t")

            # save models and intermediate checkpoints
            if step % args.checkpoint_steps == 1 or step == args.train_steps:
                torch.save(encoder_img.state_dict(), os.path.join(args.save_dir, "encoder_img.pt"))
                torch.save(encoder_txt.state_dict(), os.path.join(args.save_dir, "encoder_txt.pt"))
                if args.save_all_checkpoints:
                    torch.save(encoder_img.state_dict(), os.path.join(args.save_dir, "encoder_img_%d.pt" % step))
                    torch.save(encoder_txt.state_dict(), os.path.join(args.save_dir, "encoder_txt_%d.pt" % step))
            step += 1

    # evaluation
    # ----------
    
    # collect encodings and labels from the validation and test data
    val_dict = get_data(val_dataset, encoder_img, encoder_txt, loss_func, dataloader_kwargs)
    test_dict = get_data(test_dataset, encoder_img, encoder_txt, loss_func, dataloader_kwargs)

    # print average loss values
    print(f"<Val Loss>: {np.mean(val_dict['loss_values']):.4f}")
    print(f"<Test Loss>: {np.mean(test_dict['loss_values']):.4f}")

    # handle edge case when the encodings are 1-dimensional
    if args.encoding_size == 1:
        for m in ["image", "text"]:
            val_dict[f"hz_{m}"] = val_dict[f"hz_{m}"].reshape(-1, 1)
            test_dict[f"hz_{m}"] = test_dict[f"hz_{m}"].reshape(-1, 1)

    # standardize the encodings
    for m in ["image", "text"]:
        scaler = StandardScaler()
        val_dict[f"hz_{m}"] = scaler.fit_transform(val_dict[f"hz_{m}"])
        test_dict[f"hz_{m}"] = scaler.transform(test_dict[f"hz_{m}"])

    # evaluate how well each factor can be predicted from the encodings
    results = []
    semantics = MultimodalMPI3DRealComplex.VAL_LATENTS     # full visual latents
    for m in ["image", "text"]:
        for ix, semantic_name in enumerate(semantics):

            # select data
            train_inputs = val_dict[f"hz_{m}"]
            test_inputs = test_dict[f"hz_{m}"]
            train_labels = val_dict[f"semantics"][semantic_name]
            test_labels = test_dict[f"semantics"][semantic_name]

            data = [train_inputs, train_labels, test_inputs, test_labels]
            mcc_logreg, mcc_mlp = [np.nan] * 2

            # logistic classification
            logreg = LogisticRegression(n_jobs=-1, max_iter=10000)
            mcc_logreg = evaluate_prediction(logreg, matthews_corrcoef, *data)
            # nonlinear classification
            mlpreg = MLPClassifier(max_iter=10000)
            mcc_mlp = evaluate_prediction(mlpreg, matthews_corrcoef, *data)

            # append results
            results.append([ix, m, semantic_name, mcc_logreg, mcc_mlp])

    # convert evaluation results into tabular form
    columns = ["ix", "modality", "semantic_name", "mcc_logreg", "mcc_mlp"]
    df_results = pd.DataFrame(results, columns=columns)
    df_results.to_csv(os.path.join(args.save_dir, "results.csv"))
    print(df_results.to_string())


if __name__ == "__main__":
    main()
