from typing import Optional

import torch
import torch.nn.functional as F


def calculate_recall(img: torch.Tensor,
                     targets: torch.Tensor,
                     masks: torch.Tensor,
                     W_u_i: set[int],
                     R_x: torch.Tensor,
                     threshold_iou: float) -> Optional[float]:
    """
    Calculate the recall for a single image.

    Args:
        img (torch.Tensor): The image on which to compute the recall.
            Shape: [3, 224, 224].
        targets (torch.Tensor): The ground-truth targets for the image.
            Shape: [num_instances].
        masks (torch.Tensor): The ground-truth masks for the image.
            Shape: [num_instances, 1, filter_width, filter_height].
        W_u_i (set[int]): The GloVe indices of the top `s` predicted words.
        R_x (torch.Tensor): The area to be used for IoU calculation.
            Shape: [1, 224, 224].
        threshold_iou (float): The threshold for IoU.

    Returns:
        Optional[float]: The recall. If there are no ground-truth words, None
            is returned.
    """
    # Get mask to be used for IoU calculation.
    masks_img_resized = F.interpolate(masks,
                                      size=img.shape[-2:],
                                      mode="nearest").bool()

    # Calculate the ground-truth words.
    G_u_i = set()
    for M_j, t_j in zip(masks_img_resized, targets):
        IoU = (R_x & M_j).sum() / (R_x | M_j).sum()
        if IoU > threshold_iou:
            G_u_i.add(t_j.item())

    # Compute the recall.
    if len(G_u_i) == 0:
        return None
    return len(G_u_i & W_u_i) / len(G_u_i)
