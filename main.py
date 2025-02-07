from PIL import Image, ImageSequence
import numpy as np
import torch
import torch.nn as nn

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


path_gif = "./Test/OAS1_0311_MR1_mpr-1_anon_sag_66.gif"

# El gif consta de sola una imagen, por tanto como no es un "video" no necesitaremos recorrer la secuencia
# de imagenes del gif y directamente se podría acceder a la imagen.

gif = Image.open(path_gif)

# Se convierte a RGB la imagen y se pasa a uint8

# u <-- sin signo
# int8 <-- valores en el pixel range [0, 255]
frame = np.array(gif.convert("RGB"), dtype=np.float32)
frame_tensor = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)


print(frame_tensor.shape)


Conv2d = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, stride=1, bias=False)
MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)
Relu = nn.ReLU()


# El filtro tiene 4 dimensiones, AxBxCxD
# A <- # de filtros a utilizar
# B <- # Número de canales de entrada a los que se les aplicará el filtro
# CxD <- Matriz con la estructura y valores del filtro

kernel = torch.tensor([[ 0, -1,  0],
                       [-1,  4, -1],
                       [ 0, -1,  0]], dtype=torch.float32)


# Un filtro para solo un canal, con solo una salida esperada

kernel = kernel.view(1, 1, 3, 3)

kernel = kernel.repeat(3, 3, 1, 1)

with torch.no_grad():

    # Para realizarlo óptimamente, se tiene que convertir el tensor a un parámetro de una red neuronal
    # para que obtenga todas las propiedades de la misma (autograd y otros).
    Conv2d.weight = nn.Parameter(kernel)

# Convoluciones
ImgFiltro1 = Conv2d(frame_tensor)
ImgFiltro1 = Relu(ImgFiltro1)
ImgFiltro1 = MaxPool2d(ImgFiltro1)

ImgFiltro1 = Conv2d(ImgFiltro1)
ImgFiltro1 = Relu(ImgFiltro1)
ImgFiltro1 = MaxPool2d(ImgFiltro1)

ImgFiltro1 = Conv2d(ImgFiltro1)
ImgFiltro1 = Relu(ImgFiltro1)
ImgFiltro1 = MaxPool2d(ImgFiltro1)


# Ajuste de las dimensiones.
ImgFiltro1 = ImgFiltro1.squeeze(0).permute(1, 2, 0)
ImgFiltro1 = ((ImgFiltro1 - torch.min(ImgFiltro1)) / (torch.max(ImgFiltro1) - torch.min(ImgFiltro1)) * 255)
ImgFiltro1 = np.round(ImgFiltro1.cpu().detach().numpy()).astype(np.uint8)

print(ImgFiltro1.shape)

plt.imshow(ImgFiltro1)
plt.axis("off")
plt.show()
