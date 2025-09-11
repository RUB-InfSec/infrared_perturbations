import cv2
import kornia
import numpy as np
import numpy.typing as npt
import torch
import torchvision
from tqdm import tqdm


def equalize_L(img, target_brightness, noshow=False):
    for i in tqdm(range(img.shape[0]), disable=noshow):
        img[i] = kornia.color.rgb_to_lab(img[i])
        L = torch.mean(img[i][0])
        img[i][0] = img[i][0] * (target_brightness / L)
        img[i] = kornia.color.lab_to_rgb(img[i])
    return img.clamp(0, 1)

def transform_to_ir(img, lux):
    mask = torch.ones_like(img)

    approx_r_scale = np.poly1d([4.73334986e-12, -4.86357179e-08, -4.29679842e-06, 7.61929250e-01])(lux)
    approx_g_scale = np.poly1d([5.27304477e-12, -5.24749768e-08, -3.83417677e-05, 1.00125619e+00])(lux)
    approx_b_scale = np.poly1d([8.95281729e-12, -1.02830798e-07, 1.18434268e-04, 1.14222390e+00])(lux)

    for i in range(img.shape[0]):
        pic_vi = img[i].clone() * torch.where(mask[i] == 0, 0, 1)
        pic_vi[0] = pic_vi[0] + (pic_vi[0] * approx_r_scale)
        pic_vi[1] = pic_vi[1] + (pic_vi[0] * approx_g_scale)
        pic_vi[2] = pic_vi[2] + (pic_vi[0] * approx_b_scale)
        img[i] = (pic_vi * torch.where(mask[i] == 0, 0, 1))

    # clip output image into valid range
    return img.clamp(0, 1)


TT = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

def to_tensor_batch(imgs):
    img_out = torch.zeros((imgs.shape[0], imgs.shape[3], imgs.shape[1], imgs.shape[2]))
    for i in range(len(imgs)):
        img_out[i] = TT(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
    return img_out


def back_transform(imgs):
    img_out = np.zeros((imgs.shape[0], imgs.shape[2], imgs.shape[3], imgs.shape[1]), dtype=np.uint8)
    for i in range(len(imgs)):
        img_out[i] = cv2.cvtColor(np.asarray(torchvision.transforms.functional.to_pil_image(imgs[i])),
                                  cv2.COLOR_RGB2BGR)
    return img_out


def get_L(img):
    return torch.mean(torch.mean(kornia.color.rgb_to_lab(img)[:, 0], dim=1), dim=1)


def equalize_and_transform_to_ir(images: npt.NDArray, lux: float):
    # Compute the average L value for the whole dataset
    L_mean = torch.mean(get_L(to_tensor_batch(images)))

    # Simulate the effect of IR light for each image in the dataset
    images_ir = to_tensor_batch(images)
    images_ir = transform_to_ir(equalize_L(images_ir, L_mean, True), lux)
    images_ir = back_transform(images_ir)

    # Normalize the dataset's brightness
    images = to_tensor_batch(images)
    images = equalize_L(images, L_mean, True)
    images = back_transform(images)
    return images, images_ir


def equalize_data(images: npt.NDArray):
    # Compute the average y value for the whole dataset
    L_mean = torch.mean(get_L(to_tensor_batch(images)))

    # Normalize the dataset's brightness
    images = to_tensor_batch(images)
    images = equalize_L(images, L_mean, True)
    images = back_transform(images)
    return images

