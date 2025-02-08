import numpy as np
import torch.nn as nn
import torch


def Convolution(Images, NumConvs, Conv2d):
    MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)
    Relu = nn.ReLU()

    # Xsiaca, por si al usar ese contenido en la función, altera al array original
    ConvolutionImage = Images

    for i in range(NumConvs):
        ConvolutionImage = Conv2d(ConvolutionImage)
        ConvolutionImage = Relu(ConvolutionImage)
        ConvolutionImage = MaxPool2d(ConvolutionImage)

    return ConvolutionImage


def ConvolutionToPixelRange(ConvolutionImages):

    # Ajustando las dimensiones, a las que eran originalmente para plotearlas ;b
    ImagesFilter = ConvolutionImages.permute(0, 2, 3, 1)

    normalized_images = []
    for i in range(ImagesFilter.shape[0]):  # Iteramos sobre las 4 imágenes
        img = ImagesFilter[i]  # Seleccionamos la imagen i
        img = (img - img.amin()) / (img.amax() - img.amin()) * 255  # Normalizamos
        normalized_images.append(img)  # Guardamos la imagen normalizada

    ImagesFilter = torch.stack(normalized_images)  # Convertimos la lista en un tensor nuevamente
    ImagesFilter = np.round(ImagesFilter.cpu().detach().numpy()).astype(np.uint8)

    return ImagesFilter