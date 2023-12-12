import argparse
import warnings

import torch
import torch.nn as nn
import torchvision


def setup_explainer(args: argparse.Namespace,
                    random_feature: bool = False) -> nn.Module:
    """
    This function is used to set up the explainer model.

    Args:
        args (argparse.Namespace): The command line arguments.
        random_feature (bool, optional): Whether to use randomly initialized
            models instead of pretrained feature extractors. Defaults to False.

    Returns:
        nn.Module: Feat() + Exp() model.
    """
    if random_feature:
        # Load the model without pretrained weights.
        model = torchvision.models.__dict__[args.model](weights=None)
    else:
        # Load the model with pretrained weights.
        # The argument "pretrained" is depricated, but there is no way to
        # specify the "weights" argument since the args.model string does not
        # correspond in general to the upper- and lowercase letters used in
        # torchvision.models. Therefore, we ignore the warning thrown here.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            model = torchvision.models.__dict__[args.model](pretrained=True)

    # Freeze the weights of the target model.
    for param in model.parameters():
        param.requires_grad = False

    # Get the index of the target layer (e.g. layer4).
    target_index = list(model._modules).index(args.layer_target)

    # Get the index of the classifier layer (e.g. fc).
    classifier_index = list(model._modules).index(args.layer_classifier)

    # Replace the layers between the target and classifier layer with identity
    # layers. This is done to avoid unnecessary computation of these layers.
    for module_name in list(model._modules)[target_index + 1:classifier_index]:
        if not module_name.endswith("pool"):
            model._modules[module_name] = nn.Identity()

    # Get the feature dimension of the classifier layer.
    if args.model.startswith("resnet"):
        feature_dim = model._modules[args.layer_classifier].in_features
    elif args.model.startswith("alexnet"):
        feature_dim = model._modules[args.layer_classifier]._modules["1"] \
            .in_features
    else:
        raise NotImplementedError(f"Model '{args.model}' is not supported.")

    # Replace the classifier layer with our Feature Explainer model.
    model._modules[args.layer_classifier] = nn.Sequential(
        nn.BatchNorm1d(feature_dim),
        nn.Dropout(0.1),
        nn.Linear(in_features=feature_dim,
                  out_features=feature_dim,
                  bias=True),
        nn.ReLU(),
        nn.BatchNorm1d(feature_dim),
        nn.Dropout(0.1),
        nn.Linear(in_features=feature_dim,
                  out_features=args.word_embedding_dim,
                  bias=True)
    )

    # Move the feature extracter model (Feat) and the explainer model (Exp),
    # which are now combined into `model`, to the GPU.
    model.cuda()

    return model


def forward_Feat(args: argparse.Namespace, model: nn.Module,
                 imgs: torch.Tensor) -> torch.Tensor:
    """
    Perform a forward pass through the feature extractor Feat().

    Args:
        args (argparse.Namespace): The command line arguments.
        model (nn.Module): Feat() + Exp() model.
        imgs (torch.Tensor): Batch of input images.
            Shape: [batch_size, 3, 224, 224].

    Returns:
        torch.Tensor: Batch of filter activations.
            Shape: [batch_size, amount_filters, filter_width, filter_height].
    """
    for name, module in model._modules.items():
        imgs = module(imgs)
        if name == args.layer_target:
            return imgs


def forward_Exp(args: argparse.Namespace, model: nn.Module,
                acts: torch.Tensor) -> torch.Tensor:
    """
    Perform a forward pass through the feature explainer Exp().

    Args:
        args (argparse.Namespace): The command line arguments.
        model (nn.Module): Feat() + Exp() model.
        acts (torch.Tensor): Batch of filter activations.
            Shape: [batch_size, amount_filters, filter_width, filter_height].

    Returns:
        torch.Tensor: Batch of word embeddings.
            Shape: [batch_size, word_embedding_dim].
    """
    layer_target_seen = False
    for name, module in model._modules.items():
        if layer_target_seen:
            if name == args.layer_classifier:
                acts = torch.flatten(acts, start_dim=1)
            acts = module(acts)
        elif name == args.layer_target:
            layer_target_seen = True
    return acts
