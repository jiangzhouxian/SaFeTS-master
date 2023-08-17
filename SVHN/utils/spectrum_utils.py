import numpy as np
import torch
import torch.nn.functional as F
from torch.fft import fft2

#将numpy数组转成灰度的tensor
def np_to_gray(images):
    # 将 NumPy 数组转换为 PyTorch 张量
    images = torch.from_numpy(images)
    # 将 RGB 图像转换为灰度图像
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    gray_images = (images * weights).sum(dim=1, keepdim=True)
    return gray_images

#计算相位谱
def phase_spectra(images):
    gray_images = np_to_gray(images)
    # 应用傅立叶变换以获得频谱
    fft_images = torch.fft.fftn(gray_images, dim=(-2, -1))
    # 计算相位谱
    phase_spectrum = torch.angle(fft_images)
    # 将 phase_spectrum 张量的形状更改为 (10000, 1, 32, 32)
    return phase_spectrum

#计算高通滤波
def high_pass_filter(images):
    images = np_to_gray(images)
    # 定义 Sobel 滤波器
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # 应用高通滤波器
    filtered_x = F.conv2d(images, sobel_x, padding=1)
    filtered_y = F.conv2d(images, sobel_y, padding=1)

    # 计算梯度幅度
    saliency = torch.sqrt(filtered_x ** 2 + filtered_y ** 2)
    return saliency


#计算残差谱
def spectral_residual_saliency(images):
    gray_images = np_to_gray(images)
    saliency_maps = []
    for gray_image in gray_images:
        # 傅里叶变换
        fft_result = torch.fft.fftn(gray_image, dim=(-2, -1))

        # 计算振幅谱和相位谱
        amplitude_spectrum = torch.abs(fft_result)
        phase_spectrum = torch.angle(fft_result)

        # 计算谱残差
        log_amplitude_spectrum = torch.log1p(amplitude_spectrum)
        spectral_residual = log_amplitude_spectrum - F.avg_pool2d(log_amplitude_spectrum, 3, stride=1, padding=1)

        # 计算逆傅里叶变换
        saliency_map = torch.fft.ifftn(torch.exp(spectral_residual) * torch.exp(1j * phase_spectrum)).real
        saliency_maps.append(saliency_map)

    saliency = torch.stack(saliency_maps)
    return saliency

#计算四元Fourier的显著图
def quaternion_fourier_saliency(images):
    saliency_maps = []
    images = images.astype(np.float32)
    images = torch.from_numpy(images)
    for image in images:
        r, g, b = image[0], image[1], image[2]
        #r, g, b = image.split(dim=-3)
        R = r - (g + b) / 2
        G = g - (r + b) / 2
        B = b - (r + g) / 2
        Y = (r + g) / 2 - torch.abs(r - g) / 2 - b

        RG = R - G
        BY = B - Y
        I1 = (r + g + b) / 3

        M = torch.zeros_like(I1)

        f1 = M + RG * 1j
        f2 = BY + I1 * 1j

        F1 = torch.fft.fftn(f1)
        F2 = torch.fft.fftn(f2)

        phaseQ1 = torch.angle(F1)
        phaseQ2 = torch.angle(F2)

        ifftq1 = torch.fft.ifftn(torch.exp(phaseQ1 * 1j))
        ifftq2 = torch.fft.ifftn(torch.exp(phaseQ2 * 1j))

        absq1 = torch.abs(ifftq1)
        absq2 = torch.abs(ifftq2)

        squareq = (absq1 + absq2) * (absq1 + absq2)

        L = torch.tensor([
            [1, 4, 7, 4, 1],
            [4, 16, 26, 16, 4],
            [7, 26, 41, 26, 7],
            [4, 16, 26, 16, 4],
            [1, 4, 7, 4, 1]
        ], dtype=torch.float32) / 273
        L = L.unsqueeze(0).unsqueeze(0)

        Squareq = F.conv2d(squareq.unsqueeze(0).unsqueeze(0), L, padding=2)
        qpftmap = (Squareq - Squareq.min()) / (Squareq.max() - Squareq.min())
        saliency_maps.append(qpftmap.squeeze())

    saliency = torch.stack(saliency_maps)
    return saliency

def spec_type(images,selec_type):
    if selec_type == 'phase':
        return phase_spectra(images)
    elif selec_type == 'orig':
        return torch.from_numpy(images)
    elif selec_type == 'highpass':
        return high_pass_filter(images)
    elif selec_type == 'residual':
        return spectral_residual_saliency(images)
    elif selec_type == 'quaternion_fourier':
        return quaternion_fourier_saliency(images)
