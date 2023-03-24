from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.spatial.distance import minkowski

def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = (
        _1D_window.mm(_2D_window.reshape(1, -1))
        .reshape(window_size, window_size, window_size)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )
    window = Variable(
        _3D_window.expand(
            channel, 1, window_size, window_size, window_size
        ).contiguous()
    )
    return window

def ssim_map_3D(y, y_pred, window_size=11, size_average=True):
    img1 = torch.from_numpy(y)[None, :, :, :, :]  # add a dimension
    img2 = torch.from_numpy(y_pred)[None, :, :, :, :]
    (_, channel, _, _, _) = img1.shape
    window = create_window_3D(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map[0][0]

def mse_in_mask(X, Y, mask):
    return ((X - Y)**2).mean(where=mask.astype(bool))

def ssim_in_mask(X, Y, mask):
    ssim_map = ssim_map_3D(X, Y).numpy()
    return ssim_map.mean(where=mask.astype(bool))

def minkowski_distance(X, Y):
    return minkowski(X, Y, p=10)
