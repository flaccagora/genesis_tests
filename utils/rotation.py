import torch

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


"""REMEMBER TO ALWAYS ROTATE FROM A REFERENCE FRAME POSITION
OTHERWISE THE ROTATION WILL ACCUMULATE ERRORS"""
