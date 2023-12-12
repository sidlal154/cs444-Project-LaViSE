import torch
import torch.nn as nn


def set_bn_eval(m: nn.Module):
    """
    Set the BatchNorm layer to eval mode.

    Args:
        m (nn.Module): The module to be set to eval mode.
    """
    classname = m.__class__.__name__
    if "BatchNorm2d" in classname or "BatchNorm1d" in classname:
        m.eval()


class CSMRLoss(nn.Module):
    """Cosine Similarity Margin Ranking Loss."""

    def __init__(self, margin: float = 1.0):
        """
        Set up a Cosine Similarity Margin Ranking Loss instance.

        Args:
            margin (float, optional): The margin to be used in the loss.
                Defaults to 1.0.
        """
        super(CSMRLoss, self).__init__()
        self.margin = torch.tensor(margin).cuda()

    def forward(self, cosine_similarity: torch.Tensor, targets: torch.Tensor) \
            -> torch.Tensor:
        """
        Forward pass of the Cosine Similarity Margin Ranking Loss.
        This is an algorithmically efficient implementation of the mathematical
        loss formulation described in the paper.

        Args:
            cosine_similarity (torch.Tensor): The cosine similarity between
                each prediction and each ground-truth category embedding.
                Shape: [batch_size, num_categories].
            targets (torch.Tensor): The multiple-hot target category vector.
                Shape: [batch_size, num_categories].

        Returns:
            torch.Tensor: The average loss per sample in the batch.
        """
        # Calculate the similarity between the positive concept and the
        # prediction (i.e. dot product between v_t_j and v_hat) per sample.
        # Divide by the number of positive concepts per sample (if we are using
        # multiple-hot vectors).
        pos_concept_sim = torch.sum(targets * cosine_similarity, dim=1) \
            / torch.sum(targets, dim=1)

        # Calculate the similarity between each negative concept and the
        # prediction (i.e. dot products between v_c and v_hat for all c in C
        # where c =/= t_j) per sample.
        neg_concept_sim = (1 - targets) * cosine_similarity
        loss = (1 - targets) \
            * (self.margin - pos_concept_sim.unsqueeze(1) + neg_concept_sim)

        # Prevent NaNs (created via division by zero) from propagating.
        loss[torch.isnan(loss)] = 0

        # Calculate the final loss per sample and average over the batch.
        loss = torch.max(torch.tensor(0).cuda(), loss.float())
        loss = torch.sum(loss, dim=1)
        loss = torch.mean(loss)

        return loss
