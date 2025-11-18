import torch

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
    if center is not None:
        com = center
    else:   
        com = pos.mean(dim=1)
    pos_centered = pos - com
    pos_rotated = pos_centered @ R.T + com
    entity.set_position(pos_rotated.sceneless())

def quat_mul(q1, q2):
    """
    Multiply two quaternions.
    q1, q2: (..., 4) tensors in (w, x, y, z) format.
    Returns: (..., 4)
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    w = w2*w1 - x2*x1 - y2*y1 - z2*z1
    x = w2*x1 + x2*w1 + y2*z1 - z2*y1
    y = w2*y1 - x2*z1 + y2*w1 + z2*x1
    z = w2*z1 + x2*y1 - y2*x1 + z2*w1

    return torch.stack((w, x, y, z), dim=-1)

def rotate_rigid_entity(entity, rx, ry=None, rz=None, center=None):
    if rx.shape == torch.Size([1,3]):
        R = euler_to_quaternion(rx[0,0], rx[0,1],rx[0,2])
    elif (rx is not None) and (ry is not None) and (rz is not None):
        R = euler_to_quaternion(rx, ry,rz)
    else:
        raise ValueError

    quat = entity.get_quat()

    entity.set_quat(quat_mul(quat,R))


"""REMEMBER TO ALWAYS ROTATE FROM A REFERENCE FRAME POSITION
OTHERWISE THE ROTATION WILL ACCUMULATE ERRORS"""
