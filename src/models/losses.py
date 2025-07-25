import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def dice_loss(pred, target, eps=1e-6):
    """Computes DICE loss between predicted and target binary masks."""
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="none"):
    """Focal loss for binary classification.
    Inputs and targets must be of the same shape.
    """
    prob = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def match_predictions_to_targets(pred_masks, target_masks):
    """Match predicted masks to ground truth masks using the Hungarian algorithm."""
    pred = pred_masks.sigmoid()
    cost_matrix = torch.cdist(pred, target_masks, p=1).detach().cpu().numpy()
    pred_idx, target_idx = linear_sum_assignment(cost_matrix)
    return torch.as_tensor(pred_idx), torch.as_tensor(target_idx)


def transformer_instance_loss(pred_mask_logits, target_masks):
    """
    pred_mask_logits: [N_points, num_queries]
    target_masks: [N_points] with values in [0, num_instances - 1]
    """

    # The maximum value in target_masks corresponds to the highest label,
    # which represents the last instance. +1 because masks are 0-indexed.
    num_instances = target_masks.max().item() + 1

    # Convert ground truth into one-hot binary masks: [num_instances, N_points]
    target_onehot = torch.stack([(target_masks == i).float() for i in range(num_instances)])
    pred_masks = pred_mask_logits.permute(1, 0)  # [num_queries, N_points]

    # Match predicted masks to target masks using Hungarian algorithm
    pred_idx, target_idx = match_predictions_to_targets(pred_masks, target_onehot)

    matched_pred = pred_masks[pred_idx]  # [num_instances, N_points]
    matched_target = target_onehot[target_idx]  # [num_instances, N_points]

    loss_dice = dice_loss(torch.sigmoid(matched_pred), matched_target).mean()
    loss_focal = sigmoid_focal_loss(matched_pred, matched_target, reduction="mean")

    return loss_dice + loss_focal


def discriminative_loss(embeddings, instance_labels, delta_var=0.5, delta_dist=1.5):
    unique_labels = instance_labels.unique()
    losses = []

    centroids = []
    for label in unique_labels:
        mask = instance_labels == label
        if mask.sum() == 0:
            continue
        centroid = embeddings[mask].mean(dim=0)
        centroids.append(centroid)
        var_loss = torch.mean(F.relu(torch.norm(embeddings[mask] - centroid, dim=1) - delta_var) ** 2)
        losses.append(var_loss)

    if len(centroids) > 1:
        centroids = torch.stack(centroids)
        dists = torch.cdist(centroids, centroids, p=2)
        eye = torch.eye(dists.size(0), device=dists.device)
        dist_loss = torch.mean(F.relu(delta_dist - dists + eye * 1e5) ** 2)
        losses.append(dist_loss)

    return sum(losses)
