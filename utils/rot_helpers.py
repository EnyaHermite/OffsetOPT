import torch


def rot_x(angle):
    if not isinstance(angle, torch.Tensor):
        angle = torch.tensor(angle)
    assert angle.numel()==1

    cosval = torch.cos(angle)
    sinval = torch.sin(angle)
    val0 = torch.zeros_like(cosval)
    val1 = torch.ones_like(cosval)
    R = torch.stack([val1, val0, val0,
                     val0, cosval, -sinval,
                     val0, sinval, cosval], dim=1)
    R = torch.reshape(R, (3, 3))
    return R


def rot_y(angle):
    if not isinstance(angle, torch.Tensor):
        angle = torch.tensor(angle)
    assert angle.numel()==1

    cosval = torch.cos(angle)
    sinval = torch.sin(angle)
    val0 = torch.zeros_like(cosval)
    val1 = torch.ones_like(cosval)
    R = torch.stack([cosval, val0, sinval,
                       val0, val1, val0,
                     -sinval, val0, cosval], dim=1)
    R = R.view(3, 3)
    return R


def rot_z(angle):
    if not isinstance(angle, torch.Tensor):
        angle = torch.tensor(angle)
    assert angle.numel()==1

    cosval = torch.cos(angle)
    sinval = torch.sin(angle)
    val0 = torch.zeros_like(cosval)
    val1 = torch.ones_like(cosval)
    R = torch.stack([cosval, -sinval, val0,
                     sinval, cosval, val0,
                     val0, val0, val1], dim=1)
    R = R.view(3, 3)
    return R

