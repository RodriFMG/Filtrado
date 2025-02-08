import os

import torch
import torch.nn as nn
from ImagesPathsToTensor import ImagesPathToTensor
from ImageConvolution import Convolution, ConvolutionToPixelRange

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# Se debe colocar la dirección del directorio descargado de OASIS1. (maybe funciona para los demás OASIS xd)
path = "C:/Users/RODRIGO/Documents/disc9"

TensorListImagesMatrix = ImagesPathToTensor(path)

# 28 directorios
# 4 imagenes
# Matriz de 256x256 por cada canal (red green blue)
# 3 # de canales (RGB)

print(TensorListImagesMatrix.shape)

# Aprovecharemos los métodos brindados por torch.nn, cambiaremos el filtro a utilizar para que sea uno definido,
# el cual será uno que permita suavizar y no perder mucha información al reducir las dimensiones bruscamente.

# Se inicializa el filtro de forma aleatoria.
Conv2d = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, stride=1, bias=False)

# El filtro (weight) tiene 4 dimensiones, AxBxCxD
# A <- # de filtros a utilizar
# B <- # Número de canales de entrada a los que se les aplicará el filtro
# CxD <- Matriz con la estructura y valores del filtro

# Filtro a utilizar --> Filtro Gaussiano (según gepeto xd)
kernel = torch.tensor([[1,  2,  1],
                       [2,  4,  2],
                       [1,  2,  1]], dtype=torch.float32) / 16

# Un filtro para solo un canal, con solo una salida esperada

# Ajustamos para que se realice en todos los canales.

# tensor.view <- Redimensiona y Reorganiza todo el contenido del tensor sin perder información
# el cual, para que funcione correctamente, el número de dimensiones colocadas debe conseguir con el
# número de elementos en el tensor.
kernel = kernel.view(1, 1, 3, 3)

# Repite el contenido de las dimensiones, se busca para que el filtro sea 3x3x3x3
kernel = kernel.repeat(3, 3, 1, 1)

with torch.no_grad():
    Conv2d.weight = nn.Parameter(kernel)

# Redimensionando correctamente las dimensiones para el proceso de convolución, para este proceso
# se espera las dimensiones de:

# AxBxCxD
# A <- # de imágenes a usar
# B <- # Canales
# CxD <- # Matriz por canal.

# Permite cambiar las dimensiones (re organizando el contenido del array) en base a las dimensiones.
TensorListImagesMatrix = TensorListImagesMatrix.permute(0, 1, 4, 2, 3)

print(TensorListImagesMatrix.shape)


for NumDic, ListImagenByDic in enumerate(TensorListImagesMatrix):

    directory = f"Imagenes/OAS1_0{310+NumDic}_MR1"
    Filters32x32 = directory+"/32x32"
    Filters16x16 = directory+"/16x16"

    os.makedirs(directory, exist_ok=True)
    os.makedirs(Filters32x32, exist_ok=True)
    os.makedirs(Filters16x16, exist_ok=True)

    ConvolutionImages3 = Convolution(Images=TensorListImagesMatrix[NumDic], NumConvs=3, Conv2d=Conv2d)
    ImagesFilter3 = ConvolutionToPixelRange(ConvolutionImages3)

    ConvolutionImages4 = Convolution(Images=TensorListImagesMatrix[NumDic], NumConvs=4, Conv2d=Conv2d)
    ImagesFilter4 = ConvolutionToPixelRange(ConvolutionImages4)

    for NumImg in range(len(ImagesFilter3)):
        FilePath32x32 = Filters32x32+f"/OAS1_0{310+NumDic}_MR1_mpr-{NumImg+1}_anon_sag_66.jpg"
        FilePath16x16 = Filters16x16+f"/OAS1_0{310+NumDic}_MR1_mpr-{NumImg+1}_anon_sag_66.jpg"

        plt.imsave(FilePath32x32, ImagesFilter3[NumImg])
        plt.imsave(FilePath16x16, ImagesFilter4[NumImg])

