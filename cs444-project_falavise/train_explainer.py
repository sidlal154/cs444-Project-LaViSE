import argparse
import os
import pathlib
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from dadapt_adam import DAdaptAdam
from image_datasets import (CocoInstances, VisualGenomeInstances,
                            data_transforms)
from model_loader import forward_Exp, forward_Feat, setup_explainer
from torch.linalg import vector_norm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.vocab import GloVe
import torchtext.vocab as vocab
from train_helpers import CSMRLoss, set_bn_eval


def set_seed(seed: int):
    """
    Set a seed for all random number generators.

    Args:
        seed (int): The seed to use.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(args: argparse.Namespace,
                    model: nn.Module,
                    loss_fn: nn.Module,
                    train_loader: DataLoader,
                    embeddings: torch.Tensor,
                    epoch: int,
                    optimizer: torch.optim.Optimizer,
                    ks: list[int] = [1, 5, 10, 20]) -> Tuple[float, float]:
    """
    Train one epoch of the model.

    Args:
        args (argparse.Namespace): The command line arguments.
        model (nn.Module): Feat() + Exp() model.
        loss_fn (nn.Module): The loss function to use.
        train_loader (DataLoader): The training set dataloader.
        embeddings (torch.Tensor): The ground-truth category embeddings.
            Shape: [word_embedding_dim, num_categories].
        epoch (int): The current epoch.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        ks (list[int], optional): List of k-values. Each k represents the
            number of top category predictions to consider for accuracy
            calculation. Defaults to [1, 5, 10, 20].

    Returns:
        Tuple:
            float: The average training loss per sample.
            float: The average training accuracy@1 per sample.
    """
    # Set model to training mode.
    model.train()
    model.apply(set_bn_eval)

    # Initialize variables.
    ks = sorted(set(ks).union((1,)))
    amount_batches = len(train_loader)
    avg_loss = 0
    avg_acc = {k: 0 for k in ks}

    # Iterate over the training set.
    for batch_idx, (imgs, targets, masks) in enumerate(train_loader):
        # Move batch data to GPU.
        imgs, targets, masks = imgs.cuda(), targets.cuda(), masks.cuda()
        curr_batch_size = imgs.shape[0]

        # Forward pass.
        acts = forward_Feat(args, model, imgs)
        if torch.sum(masks) > 0:
            acts *= masks
        preds = forward_Exp(args, model, acts)

        # Compute the cosine similarity between each prediction and each
        # ground-truth category embedding.
        cosine_similarity = (preds @ embeddings) \
            / (vector_norm(preds, dim=1, keepdim=True) @
               vector_norm(embeddings, dim=0, keepdim=True))

        # Calculate loss.
        loss = loss_fn(cosine_similarity, targets)

        # Backward propagation.
        optimizer.zero_grad()
        if loss.requires_grad:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            # Update average loss.
            avg_loss += loss.item() * curr_batch_size

            # Sort the results in descending order, so that the top `k` indices
            # represent the categories that are most similar to the model's
            # prediction.
            cat_preds_per_sample = torch.argsort(cosine_similarity, dim=1,
                                                 descending=True)[:, :max(ks)]

            # Update average accuracy.
            for img_idx, cat_preds in enumerate(cat_preds_per_sample):
                for k in ks:
                    avg_acc[k] += targets[img_idx, cat_preds[:k]].any().item()

            # Print logging data.
            if (batch_idx + 1) % 10 == 0 or batch_idx == amount_batches - 1:
                epoch_chars = len(str(args.epochs))
                batch_chars = len(str(amount_batches))
                print(f"[Epoch {epoch:{epoch_chars}d}: "
                      f"{batch_idx + 1:{batch_chars}d}/{amount_batches} "
                      f"({int((batch_idx + 1) / amount_batches * 100):3d}%)] "
                      f"Loss: {loss.item():.4f}")
                if args.wandb:
                    wandb.log({"Iter_Train_Loss": loss})

            # Save checkpoint.
            if (batch_idx + 1) % args.save_every == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                    },
                    os.path.join(args.dir_save, "ckpt_tmp.pth.tar")
                )

        # Free memory.
        torch.cuda.empty_cache()

    # Calculate average loss and average accuracy per sample.
    avg_loss /= len(train_loader.dataset)
    avg_acc = {k: avg_acc[k] / len(train_loader.dataset) * 100
               for k in ks}
    print()
    print(f"Train average loss: {avg_loss:.4f}")
    for k in ks:
        print(f"Train top-{k} accuracy: {avg_acc[k]:.2f}%")

    return avg_loss, avg_acc[1]


def validate(args: argparse.Namespace,
             model: nn.Module,
             loss_fn: nn.Module,
             valid_loader: DataLoader,
             embeddings: torch.Tensor,
             ks: list[int] = [1, 5, 10, 20]) -> Tuple[float, float]:
    """
    Validate the model.

    Args:
        args (argparse.Namespace): The command line arguments.
        model (nn.Module): Feat() + Exp() model.
        loss_fn (nn.Module): The loss function to use.
        valid_loader (DataLoader): The validation set data loader.
        embeddings (torch.Tensor): The ground-truth category embeddings.
            Shape: [word_embedding_dim, num_categories].
        ks (list[int], optional): List of k-values. Each k represents the
            number of top category predictions to consider for accuracy
            calculation. Defaults to [1, 5, 10, 20].

    Returns:
        Tuple:
            float: The average validation loss per sample.
            float: The average validation accuracy@1 per sample.
    """
    # Set model to evaluation mode.
    model.eval()

    # Initialize variables.
    ks = sorted(set(ks).union((1,)))
    avg_loss = 0
    avg_acc = {k: 0 for k in ks}

    # Iterate over the validation set.
    for imgs, targets, masks in valid_loader:
        with torch.no_grad():
            # Move batch data to GPU.
            imgs, targets, masks = imgs.cuda(), targets.cuda(), masks.cuda()
            curr_batch_size = imgs.shape[0]

            # Forward pass.
            acts = forward_Feat(args, model, imgs)
            if torch.sum(masks) > 0:
                acts *= masks
            preds = forward_Exp(args, model, acts)

            # Compute the cosine similarity between each prediction and each
            # ground-truth category embedding.
            cosine_similarity = (preds @ embeddings) \
                / (vector_norm(preds, dim=1, keepdim=True) @
                   vector_norm(embeddings, dim=0, keepdim=True))

            # Calculate loss.
            loss = loss_fn(cosine_similarity, targets)

            # Update average loss.
            avg_loss += loss.item() * curr_batch_size

            # Sort the results in descending order, so that the top `k` indices
            # represent the categories that are most similar to the model's
            # prediction.
            cat_preds_per_sample = torch.argsort(cosine_similarity, dim=1,
                                                 descending=True)[:, :max(ks)]

            # Update average accuracy.
            for img_idx, cat_preds in enumerate(cat_preds_per_sample):
                for k in ks:
                    avg_acc[k] += targets[img_idx, cat_preds[:k]].any().item()

        # Free memory.
        torch.cuda.empty_cache()

    # Calculate average loss and average accuracy per sample.
    avg_loss /= len(valid_loader.dataset)
    avg_acc = {k: avg_acc[k] / len(valid_loader.dataset) * 100
               for k in ks}
    print()
    print(f"Valid average loss: {avg_loss:.4f}")
    for k in ks:
        print(f"Valid top-{k} accuracy: {avg_acc[k]:.2f}%")

    return avg_loss, avg_acc[1]


def main(args: argparse.Namespace):
    """
    Train a model on the specified dataset.

    Args:
        args (argparse.Namespace): The command line arguments.
    """
    # Make sure the run is deterministic and reproducible.
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set up run name.
    if args.name is None:
        args.name = f"{args.model}-{args.layer_target}-imagenet-{args.refer}"
    if args.random:
        args.name += "-random"

    # Set up output path.
    args.dir_save = os.path.join(args.dir_save, args.name)
    if not os.path.exists(args.dir_save):
        os.makedirs(args.dir_save)

    # Set up wandb logging.
    if args.wandb:
        path_wandb_id_file = pathlib.Path(os.path.join(args.dir_save,
                                                       "runid.txt"))
        print(f"Creating new wandb instance at '{path_wandb_id_file}'...",
              end=" ")
        run = wandb.init(project="temporal_scale",
                         name=args.name, config=args)
        path_wandb_id_file.write_text(str(run.id))
        print("Done.")

    # Set up word embeddings.
    # [CS444: modifications to include other embeddings]:
    if args.word_model == "glove":
        word_emb = GloVe(name="6B", dim=args.word_embedding_dim)
    elif args.word_model == "fasttext":
        url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip'
        word_emb = vocab.Vectors(name='wiki-news-300d-1M.vec', url=url)
    else:
        raise NotImplementedError(f"Word embeddings '{args.word_model}' is "
                                  "not supported in this code.")
    # [CS444: mod ends]
    
    torch.cuda.empty_cache()

    # Set up model.
    model = setup_explainer(args, args.random)
    if args.path_model is None:
        args.path_model = os.path.join(args.dir_save, "ckpt_best.pth.tar")
    if os.path.exists(args.path_model):
        print(f"Loading model from '{args.path_model}'...", end=" ")
        model.load_state_dict(torch.load(args.path_model)["state_dict"])
        print("Done.")
    else:
        print(f"No model found at '{args.path_model}'. "
              "Training from scratch.")
    model = model.cuda()

    # Calculate the filter dimensions.
    _, _, filter_width, filter_height = forward_Feat(
        args, model, torch.zeros(1, 3, 224, 224).cuda()
    ).shape

    # Set up dataset.
    if args.refer == "coco":
        # Set up COCO training and validation dataset.
        root = os.path.join(args.dir_data, "coco")
        datasets = {}
        datasets["train"] = CocoInstances(
            ann_file=os.path.join(root,
                                  "annotations/instances_train2017-animal-nocrowd.json"),
            root=os.path.join(root, "train2017"),
            cat_mappings_file=os.path.join(root, "cat_mappings.pkl"),
            transform=data_transforms["train"],
            filter_width=filter_width,
            filter_height=filter_height
        )
        datasets["val"] = CocoInstances(
            ann_file=os.path.join(root,
                                  "annotations/instances_val2017-animal-nocrowd.json"),
            root=os.path.join(root, "val2017"),
            cat_mappings_file=os.path.join(root, "cat_mappings.pkl"),
            transform=data_transforms["val"],
            filter_width=filter_width,
            filter_height=filter_height
        )

        # Set up word embeddings.
        cat_mappings = datasets["train"].cat_mappings
        glove_indices = [
            glove_idx
            for _, glove_idx in sorted(cat_mappings["stoi"].items())
        ]
        embeddings = word_emb.vectors[glove_indices].T.cuda()
    else:
        raise NotImplementedError(f"Reference dataset '{args.refer}' is not "
                                  "implemented.")

    # Set up dataloader.
    dataloaders = {
        dataset_type: DataLoader(dataset, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)
        for dataset_type, dataset in datasets.items()
    }

    # Set up optimizer, scheduler and loss function.
    optimizer = DAdaptAdam(model.parameters())
    loss_fn = CSMRLoss(margin=args.margin)

    # Print confirmation message.
    print()
    print("[ Model was set up successfully! ]".center(79, "-"))
    print()

    # ------------------------------ TRAINING ------------------------------- #

    loss_valid_best = 99999999
    accs_train = []
    accs_valid = []

    with open(os.path.join(args.dir_save, "valid.txt"), "w") as f:
        for epoch in range(1, args.epochs+1):
            print()
            print(f"[ Epoch {epoch} starting ]".center(79, "-"))

            # Train and validate.
            loss_train, acc_train = train_one_epoch(args,
                                                    model,
                                                    loss_fn,
                                                    dataloaders["train"],
                                                    embeddings,
                                                    epoch,
                                                    optimizer)
            loss_valid, acc_valid = validate(args,
                                             model,
                                             loss_fn,
                                             dataloaders["val"],
                                             embeddings)

            # Setup wandb logging.
            if args.wandb:
                wandb.log({"Epoch": epoch})
                wandb.log({"Epoch": epoch, "Epoch_Ave_Train_Loss": loss_train})
                wandb.log({"Epoch": epoch, "Epoch_Ave_Train_Acc": acc_train})
                wandb.log({"Epoch": epoch, "Epoch_Ave_Valid_Loss": loss_valid})
                wandb.log({"Epoch": epoch, "Epoch_Ave_Valid_Acc": acc_valid})
                wandb.log({"Epoch": epoch,
                           "LR": optimizer.param_groups[0]["lr"]})

            # Save train and validation accuracy.
            accs_train.append(acc_train)
            accs_valid.append(acc_valid)
            # scheduler.step(loss_valid)
            f.write("epoch: %d\n" % epoch)
            f.write("train loss: %f\n" % loss_train)
            f.write("train accuracy: %f\n" % acc_train)
            f.write("valid loss: %f\n" % loss_valid)
            f.write("valid accuracy: %f\n" % acc_valid)

            # Save checkpoint if validation loss is the lowest loss so far.
            if loss_valid < loss_valid_best:
                loss_valid_best = loss_valid
                print("==> new checkpoint saved")
                f.write("==> new checkpoint saved\n")
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                    },
                    os.path.join(args.dir_save, "ckpt_best.pth.tar")
                )
                plt.figure()
                plt.plot(loss_train, "-o", label="train")
                plt.plot(loss_valid, "-o", label="valid")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend(loc="upper right")
                plt.savefig(os.path.join(args.dir_save, "losses.png"))
                plt.close()

    # Save wandb summary.
    if args.wandb:
        wandb.run.summary["best_validation_loss"] = loss_valid_best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for data loading")
    parser.add_argument("--cat-frac", type=float, default=0.7,
                        help="Fraction of categories used for VG supervision")
    parser.add_argument("--dir-data", type=str, default="./data",
                        help="Path to the datasets")
    parser.add_argument("--dir-save", type=str, default="./outputs",
                        help="Path to model checkpoints")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs")
    parser.add_argument("--layer-classifier", type=str, default="fc",
                        help="Name of classifier layer")
    parser.add_argument("--layer-target", type=str, default="layer4",
                        help="Target layer to explain")
    parser.add_argument("--margin", type=float, default=1.0,
                        help="Hyperparameter for margin ranking loss")
    parser.add_argument("--model", type=str, default="resnet18",
                        help="Target network")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of subprocesses to use for data loading")
    parser.add_argument("--path-model", type=str, default=None,
                        help="Path to trained explainer model")
    parser.add_argument("--random", action="store_true",
                        help="Use a randomly initialized target model instead "
                        "of torchvision pretrained weights")
    parser.add_argument("--refer", type=str, default="coco",
                        choices=("vg", "coco"),
                        help="Reference dataset")
    # [CS444: modifications to include other embeddings]:
    parser.add_argument("--word-model", type=str, default="glove", choices=("glove","fasttext"),
                        help="Word embeddings model to use")
    # [CS444: mod ends]:
    parser.add_argument("--save-every", type=int, default=1000,
                        help="How often to save a model checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed to use")
    parser.add_argument("--train-frac", type=float, default=0.9,
                        help="Fraction of data used for training")
    parser.add_argument("--wandb", action="store_true",
                        help="Use wandb for logging")
    parser.add_argument("--word-embedding-dim", type=int, default=300,
                        help="GloVe word embedding dimension to use")

    args = parser.parse_args()
    print(f"{args=}")

    main(args)