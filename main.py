import nibabel as nib
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # Cambia el backend a TkAgg

from MedicalFormatToPNG import MedicalFormatToPNGAndGenerator

# El hrd brinda metadatos como dimensiones del .img

# Cargamos el contenido del hdr del .img
img_info = nib.load("./Test/OAS1_0311_MR1_mpr-2_anon.hdr")

# meta datos
hdr = img_info.header
print(hdr, "\n\n")

# extraemos las dimensiones del .img
data = img_info.get_fdata()
ExpectedDimentions = data.shape

print("Dimensiones de la img estructurada:", ExpectedDimentions)

# El .img es la imagen pero puestas de forma cruda ( es como si fuera un vector unidimensional
# con el contenido puesto secuencialmente, donde la secuencia es según a las dimensiones esperadas ),
# el cual con el .hdr podremos hacer un reshape a ese contenido
# para estructurar todo ese "vector" y tener el formato de la imagen en una matriz, como tambien su data type.

# dataype : int16 <-- info de la meta data (usar este y no el data.dtype porque altera las dimmensiones
# esperadas).
dtype = np.int16

# Extraemos el contenido del formato .img
with open("./Test/OAS1_0311_MR1_mpr-2_anon.img") as UncompressedImg:
    CompressedImg = np.fromfile(UncompressedImg, dtype=dtype)

# Si nos damos cuenta, la dimensión es igual al producto 256*256*128 ( dimensiones esperadas, pero totalmente
# aplanadas ).
print("Unidimentional Vector Shape .Img", CompressedImg.shape)

# eje x (ancho), eje y (alto), eje z (volumen), ignore
[dimx, dimy, dimz, _] = ExpectedDimentions

# (volumen x alto x ancho)
MedicalImg = CompressedImg.reshape((dimz, dimy, dimx))

print(MedicalImg.shape)

MedicalPngGenerator = MedicalFormatToPNGAndGenerator(MedicalImg)

plt.imshow(MedicalPngGenerator[1], cmap='gray')
plt.axis("off")
plt.show()