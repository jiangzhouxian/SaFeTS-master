import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.fft import fft2
import matplotlib.pyplot as plt
import cv2

#计算相位谱
'''
def phase_spectra(image):
    #gray_images = np_to_gray(images).numpy()
    image = image.squeeze().numpy()
    # 应用傅立叶变换以获得频谱
    f = np.fft.fft2(image)
    #fft_images = np.fft.fftn(image, axes=(-2, -1))
    # 计算相位谱
    phase_spectrum = np.angle(f)
    # 将 phase_spectrum 张量的形状更改为 (1, 32, 32)
    i_ph = np.fft.ifft2(phase_spectrum)
    i_ph = np.abs(i_ph)
    return i_ph
'''
def phase_spectra(image):

    fft_result = torch.fft.fftn(image, dim=(-2, -1))
    phase_spectrum = torch.angle(fft_result)
    saliency_map = torch.fft.ifftn(torch.exp(1j * phase_spectrum)).real
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

    return saliency_map.squeeze()


#计算高通滤波
def high_pass_filter(image):
    #images = np_to_gray(images)
    image = image.to(torch.float32)  # Convert image to float32
    #img = img.numpy()
    # 定义 Sobel 滤波器
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # 应用高通滤波器
    filtered_x = F.conv2d(image, sobel_x, padding=1)
    filtered_y = F.conv2d(image, sobel_y, padding=1)

    # 计算梯度幅度
    saliency = torch.sqrt(filtered_x ** 2 + filtered_y ** 2)
    return saliency.squeeze()


#计算残差谱
def spectral_residual_saliency(image):
    #gray_images = np_to_gray(images)
    image = image.to(torch.float32)  # Convert image to float32
    #img = img.numpy()
    saliency_maps = []
    for gray_image in image:
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
    return saliency.squeeze()

#计算四元Fourier的显著图
def quaternion_fourier_saliency(images):
    images = images.to(torch.float32)  # Convert image to float32
    #img = img.numpy()
    saliency_maps = []
    #images = images.astype(np.float32)
    #images = torch.from_numpy(images)
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
    return saliency.squeeze()
    
def preprocess_img(image):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return preprocess(image)

def spec_type(image, selec_type):
    if selec_type == 'quaternion_fourier':
        img = cv2.imread(image)
        img = preprocess_img(img)
        img = img.unsqueeze(0) 
        return quaternion_fourier_saliency(img)
    else:
        if selec_type == 'orig':
            img = cv2.imread(image)
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return image_rgb
        else:
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = preprocess_img(img)  # Preprocess the image
            img = img.unsqueeze(0)  # Add a batch dimension to the image
            if selec_type == 'phase':
                return phase_spectra(img)
            elif selec_type == 'highpass':
                return high_pass_filter(img)
            elif selec_type == 'residual':
                return spectral_residual_saliency(img)
# Linear scaling
def linear_scaling(phase_spectrum):
    phase_min = 0
    phase_max = 1
    phase_scaled = (phase_spectrum - phase_min) / (phase_max - phase_min)
    return phase_scaled

def adaptive_histogram_equalization(phase_spectrum, clip_limit):
    clahe = cv2.createCLAHE(clipLimit=clip_limit)
    return clahe.apply(phase_spectrum)


# Histogram equalization
def histogram_equalization(phase_spectrum,bins_num):
    phase_histogram = np.histogram(phase_spectrum, bins=bins_num, range=(-np.pi, np.pi))[0]
    phase_cdf = np.cumsum(phase_histogram)
    phase_cdf_normalized = phase_cdf / phase_cdf[-1]
    phase_equalized = np.interp(phase_spectrum, np.linspace(-np.pi, np.pi, num=bins_num), phase_cdf_normalized)
    return phase_equalized

def show_img_spectrum(image):
    orig, ph, hp, rd, qft = spec_type(image, 'orig'), spec_type(image, 'phase'), spec_type(image, 'highpass'), spec_type(image, 'residual'), spec_type(image, 'quaternion_fourier')
    ph = histogram_equalization(ph,256)
    rd = histogram_equalization(rd,512)
    #rd = linear_scaling(rd)
    # Remove the batch dimension before displaying
    #orig, ph, hp, rd, qft = orig[0], ph[0], hp[0], rd[0], qft[0]
    return orig, ph, hp, rd, qft
    #print('The Spectrums of image')
    '''
    plt.subplot(151), plt.imshow(orig, 'gray'), plt.title('orig')
    plt.axis('off')
    plt.subplot(152), plt.imshow(ph, 'gray'), plt.title('PH')
    plt.axis('off')
    plt.subplot(153), plt.imshow(hp, 'gray_r'), plt.title('HP')
    plt.axis('off')
    plt.subplot(154), plt.imshow(rd, 'gray'), plt.title('RD')
    plt.axis('off')
    plt.subplot(155), plt.imshow(qft, 'gray_r'), plt.title('QFT')
    plt.axis('off')
    plt.savefig('sliency-phase.png',dpi=500)
    plt.show()
'''

        