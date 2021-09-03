import torch


def bt_original_vs_two_augs(z):
    batch_size = z.size(0)
    z_shape = (batch_size * 2, *(z.size()[2:]))
    assert len(z.size() > 2)

    if len(z.size()) == 4:
        z1 = z[:, :2, :].reshape(z_shape)
    elif len(z.size()) == 3:
        z1 = z[:, 0, :].repeat(2, 1)

    z2 = z[:, 2:, :].reshape(z_shape)
    return z1, z2


def bt_aug1_vs_aug2(z):
    # z1, z2 = z[:, 2, :], z[:, 3, :]
    z1, z2 = z[:, 1, :], z[:, 2, :]
    return z1, z2


def bt_three_way(z):
    if len(z.size()) == 4:
        z_input1, z_input2, z_aug1, z_aug2 = (
            t.squeeze() for t in z.split(1, dim=1)
        )
        z1 = torch.cat((z_input1, z_aug1, z_aug2), dim=0)
        z2 = torch.cat((z_aug2, z_input2, z_aug1), dim=0)
    elif len(z.size()) == 3:
        z_input, z_aug1, z_aug2 = (t.squeeze() for t in z.split(1, dim=1))
        z1 = torch.cat((z_input, z_aug1, z_aug2), dim=0)
        z2 = torch.cat((z_aug2, z_input, z_aug1), dim=0)
    return z1, z2


def vic_share_original(z):
    z_input, z_aug1, z_aug2 = (t.squeeze() for t in z.split(1, dim=1))
    z1 = torch.cat((z_input, z_aug1), dim=0)
    z2 = torch.cat((z_aug2, z_input), dim=0)
    return z1, z2