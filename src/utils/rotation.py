import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from torch.types import Device

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll=X, pitch=Y, yaw=Z) to quaternions.
    Rotation order: Z (yaw) -> Y (pitch) -> X (roll)
    All inputs may be scalars or tensors of matching shape.
    """
    # Half angles
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    # Quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack([w, x, y, z], dim=-1)

def rotation_matrix_xyz(rx, ry, rz):
    Rx = torch.tensor([[1, 0, 0],
                   [0, torch.cos(rx), -torch.sin(rx)],
                   [0, torch.sin(rx), torch.cos(rx)]], dtype=torch.float32)
    
    Ry = torch.tensor([[torch.cos(ry), 0, torch.sin(ry)],
                   [0, 1, 0],
                   [-torch.sin(ry), 0, torch.cos(ry)]], dtype=torch.float32)
    
    Rz = torch.tensor([[torch.cos(rz), -torch.sin(rz), 0],
                   [torch.sin(rz), torch.cos(rz), 0],
                   [0, 0, 1]], dtype=torch.float32)
    
    R = Rz @ Ry @ Rx
    return R

def rotate_entity(entity, rx, ry=None, rz=None, center=None):
    if rx.shape == torch.Size([1,3]):
        R = rotation_matrix_xyz(rx[0,0], rx[0,1],rx[0,2])
    elif ry == None or rz == None and rx.shape == torch.Size([3,3]):
        R = rx
    elif (ry is not None) and (rz is not None):
        R = rotation_matrix_xyz(rx, ry, rz)
    else:
        raise ValueError

    state = entity.get_state()
    pos = state.pos
    device = pos.device
    if center is not None:
        com = center
    else:   
        com = pos.mean(dim=1)
    pos_centered = pos - com
    pos_rotated = pos_centered @ R.T.to(device) + com
    entity.set_position(pos_rotated.sceneless())

def rotate_MPM_entity(entity, rx, ry=None, rz=None, center=None):
    if rx.shape == torch.Size([1,3]):
        R = rotation_matrix_xyz(rx[0,0], rx[0,1],rx[0,2])
    elif ry == None or rz == None and rx.shape == torch.Size([3,3]):
        R = rx
    elif (ry is not None) and (rz is not None):
        R = rotation_matrix_xyz(rx, ry, rz)
    else:
        raise ValueError

    state = entity.get_state()
    pos = state.pos
    vel = state.vel
    device = pos.device
    if center is not None:
        com = center
    else:   
        com = pos.mean(dim=1)
    pos_centered = pos - com
    pos_rotated = pos_centered @ R.T.to(device) + com
    entity.set_position(pos_rotated.sceneless())
    entity.set_velocity(np.zeros_like(vel.cpu().numpy()))

def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)

def rotate_rigid_entity(entity, rx, ry=None, rz=None, center=None):
    if rx.shape == torch.Size([1,3]):
        R = euler_to_quaternion(rx[0,0], rx[0,1],rx[0,2])
    elif rx.shape == torch.Size([3,3]):
        R = rotmat_to_quaternion(rx)
    elif (rx is not None) and (ry is not None) and (rz is not None):
        R = euler_to_quaternion(rx, ry,rz)
    else:
        raise ValueError

    quat = entity.get_quat()

    entity.set_quat(quat_mul(quat,R))


def rot6d_to_rotmat(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def rotmat_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

def generate_random_rotation_6d(batch_size=1, device='cpu'):
    """
    Generate random 6D rotation vectors.
    
    Args:
        batch_size: Number of 6D vectors to generate
        device: 'cpu' or 'cuda'
    
    Returns:
        Batch of 6D vectors (batch_size, 6)
    """
    rot_6d = torch.randn(batch_size, 6, device=device)

    if batch_size == 1:
        rot_6d = rot_6d.squeeze(0)
    
    return rot_6d


def axis_angle_to_rotmat(axis_angle):
    """
    Convert axis-angle representation to 3x3 rotation matrix using Rodrigues formula.
    Useful if you have rotation data as axis-angle.
    
    Args:
        axis_angle: Batch of axis-angle vectors (B, 3) where magnitude is angle in radians
    
    Returns:
        Batch of 3x3 rotation matrices (B, 3, 3)
    """
    batch_size = axis_angle.shape[0]
    device = axis_angle.device
    
    # Extract angle
    angle = torch.norm(axis_angle, dim=1, keepdim=True)  # (B, 1)
    
    # Normalize axis
    axis = axis_angle / (angle + 1e-8)  # (B, 3)
    
    # Rodrigues formula
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    
    # Skew-symmetric matrix of axis
    K = torch.zeros(batch_size, 3, 3, device=device)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]
    
    I = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    rot_mat = I + sin_angle.unsqueeze(2) * K + (1 - cos_angle).unsqueeze(2) * torch.bmm(K, K)
    
    return rot_mat


def euler_to_rotmat(euler_angles):
    """
    Convert Euler angles (XYZ convention) to 3x3 rotation matrix.
    
    Args:
        euler_angles: Batch of Euler angles in radians (B, 3) [roll, pitch, yaw]
    
    Returns:
        Batch of 3x3 rotation matrices (B, 3, 3)
    """
    batch_size = euler_angles.shape[0]
    device = euler_angles.device
    
    roll, pitch, yaw = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
    
    # Rotation around X axis (roll)
    Rx = torch.zeros(batch_size, 3, 3, device=device)
    Rx[:, 0, 0] = 1
    Rx[:, 1, 1] = torch.cos(roll)
    Rx[:, 1, 2] = -torch.sin(roll)
    Rx[:, 2, 1] = torch.sin(roll)
    Rx[:, 2, 2] = torch.cos(roll)
    
    # Rotation around Y axis (pitch)
    Ry = torch.zeros(batch_size, 3, 3, device=device)
    Ry[:, 0, 0] = torch.cos(pitch)
    Ry[:, 0, 2] = torch.sin(pitch)
    Ry[:, 1, 1] = 1
    Ry[:, 2, 0] = -torch.sin(pitch)
    Ry[:, 2, 2] = torch.cos(pitch)
    
    # Rotation around Z axis (yaw)
    Rz = torch.zeros(batch_size, 3, 3, device=device)
    Rz[:, 0, 0] = torch.cos(yaw)
    Rz[:, 0, 1] = -torch.sin(yaw)
    Rz[:, 1, 0] = torch.sin(yaw)
    Rz[:, 1, 1] = torch.cos(yaw)
    Rz[:, 2, 2] = 1
    
    # Combined rotation (Z * Y * X)
    rot_mat = torch.bmm(torch.bmm(Rz, Ry), Rx)
    
    return rot_mat

def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

def rotmat_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    indices = q_abs.argmax(dim=-1, keepdim=True)
    expand_dims = list(batch_dim) + [1, 4]
    gather_indices = indices.unsqueeze(-1).expand(expand_dims)
    out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
    return standardize_quaternion(out)

def quaternion_to_rotmat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def generate_random_rotation_matrix(
    n: int, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None
) -> torch.Tensor:
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(n, dtype=dtype, device=device)
    return quaternion_to_rotmat(quaternions)


def random_quaternions(
    n: int, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None
) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    if isinstance(device, str):
        device = torch.device(device)
    o = torch.randn((n, 4), dtype=dtype, device=device)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o

def verify_rotation_matrix(rot_mat, tol=1e-4):
    """
    Verify that a matrix is a valid rotation matrix.
    Should satisfy: R^T * R = I and det(R) = 1
    
    Args:
        rot_mat: 3x3 rotation matrix or batch (B, 3, 3)
        tol: tolerance for verification
    
    Returns:
        Boolean or tensor of booleans indicating validity
    """
    if rot_mat.dim() == 2:
        rot_mat = rot_mat.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Check orthogonality: R^T * R = I
    should_be_I = torch.bmm(rot_mat.transpose(1, 2), rot_mat)
    identity = torch.eye(3, device=rot_mat.device).unsqueeze(0)
    is_orthogonal = torch.allclose(should_be_I, identity, atol=tol)
    
    # Check determinant = 1
    det = torch.det(rot_mat)
    is_proper = torch.allclose(det, torch.ones_like(det), atol=tol)
    
    is_valid = is_orthogonal and is_proper
    
    if squeeze_output:
        return is_valid
    return is_valid


# Example usage
if __name__ == "__main__":
    print("=== 6D Rotation Conversion Examples ===\n")
    
    # 1. Generate random 6D vectors and convert to rotation matrices
    print("1. Random 6D → 3x3 Rotation Matrix:")
    batch_size = 1
    rot_6d = torch.randn(batch_size, 6)
    rot_mat = rot6d_to_rotmat(rot_6d)
    print(f"   Input 6D shape: {rot_6d.shape}")
    print(f"   Output 3x3 shape: {rot_mat.shape}")
    print(f"   Valid rotation? {verify_rotation_matrix(rot_mat)}")
    
    # 2. Convert back to 6D
    print("\n2. 3x3 Rotation Matrix → 6D:")
    rot_6d_recovered = rotmat_to_rot6d(rot_mat)
    print(f"   Recovered 6D shape: {rot_6d_recovered.shape}")
    print(f"   Matches original? {torch.allclose(rot_6d_recovered, rot_6d, atol=1e-5)}, expected False due 6D not unique representation")
    
    # 3. Generate random rotation matrices directly
    print("\n3. Generate random rotation matrices:")
    random_rotations = generate_random_rotation_matrix(batch_size=3)
    print(f"   Shape: {random_rotations.shape}")
    print(f"   All valid? {verify_rotation_matrix(random_rotations)}")
    
    # 4. Convert from Euler angles
    print("\n4. Euler Angles → 3x3 Rotation Matrix:")
    euler = torch.tensor([[0.5, 0.3, 0.2], [1.0, 0.5, 0.1]])  # 2 sets of Euler angles
    rot_from_euler = euler_to_rotmat(euler)
    print(f"   Input Euler shape: {euler.shape}")
    print(f"   Output 3x3 shape: {rot_from_euler.shape}")
    print(f"   Valid rotation? {verify_rotation_matrix(rot_from_euler)}")
    
    # 5. Convert from axis-angle
    print("\n5. Axis-Angle → 3x3 Rotation Matrix:")
    axis_angle = torch.randn(2, 3)
    rot_from_axis = axis_angle_to_rotmat(axis_angle)
    print(f"   Input axis-angle shape: {axis_angle.shape}")
    print(f"   Output 3x3 shape: {rot_from_axis.shape}")
    print(f"   Valid rotation? {verify_rotation_matrix(rot_from_axis)}")

    # 6. Quaternion conversions
    print("\n6. Quaternion ↔ 3x3 Rotation Matrix:")
    quat = torch.randn(2, 4)
    quat = F.normalize(quat, dim=1)  # Normalize quaternion
    rot_from_quat = quaternion_to_rotmat(quat)
    quat_recovered = rotmat_to_quaternion(rot_from_quat)
    print(f"   Input Quaternion shape: {quat.shape}")
    print(f"   Output 3x3 shape: {rot_from_quat.shape}")
    print(f"   Valid rotation? {verify_rotation_matrix(rot_from_quat)}")
    print(f"   Matches original? {torch.allclose(quat_recovered, quat, atol=1e-5)}")

    # 7. mat to 6D and back
    print("\n7. 3x3 Rotation Matrix ↔ 6D:")
    rot_mat_example = generate_random_rotation_matrix(batch_size=1)
    rot_6d_example = rotmat_to_rot6d(rot_mat_example)
    rot_mat_recovered = rot6d_to_rotmat(rot_6d_example)
    print(f"   Matches original? {torch.allclose(rot_mat_recovered, rot_mat_example, atol=1e-5)}")

    # print(rot_mat_example, "\n", rot_6d_example, "\n", rot_mat_recovered    )
    # assert verify_rotation_matrix(rot_mat_example)
    # assert verify_rotation_matrix(rot_mat_recovered)

