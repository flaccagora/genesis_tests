import torch
from utils.rotation import rot6d_to_rotmat


def geodesic_loss(rot_pred_6d, rot_target):
    """
    Geodesic loss on SO(3).
    Args:
        rot_pred_6d: predicted 6D rotation (B, 6)
        rot_target: target 3x3 rotation matrix (B, 3, 3)
    """
    batch_size = rot_pred_6d.shape[0]
    
    # Convert predictions to rotation matrices
    rot_pred_6d_expanded = rot_pred_6d.reshape(batch_size, 6)
    rot_pred = rot6d_to_rotmat(rot_pred_6d_expanded)
    
    # Frobenius norm loss on rotation matrices
    loss = torch.norm(rot_pred - rot_target, dim=[1, 2]).mean()
    
    return loss
