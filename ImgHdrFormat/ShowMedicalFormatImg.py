import nibabel as nib
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from MedicalFormatToPNG import MedicalFormatToPNGAndGenerator

# El hrd brinda metadatos como dimensiones del .img

# Cargamos el contenido del hdr del .img
img_info = nib.load("../Test/OAS1_0311_MR1_mpr-2_anon.hdr")

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
with open("../Test/OAS1_0311_MR1_mpr-2_anon.img") as UncompressedImg:
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
MedicalPngArray = np.array(list(MedicalPngGenerator))

# Obtenemos las dimensiones correctas
dimz, dimx, dimy = MedicalPngArray.shape  # Z: profundidad, X: ancho, Y: alto

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Creamos las coordenadas considerando el formato (Z, X, Y)
z = np.arange(dimz)
x = np.arange(dimx)
y = np.arange(dimy)
Z, X, Y = np.meshgrid(z, x, y, indexing="ij")  # Respetando estructura de la matriz

# Aplanamos los arrays para usarlos en scatter
values = MedicalPngArray.flatten()
Z_flat = Z.flatten()
X_flat = X.flatten()
Y_flat = Y.flatten()

# Filtramos los valores de 127 para que no se dibujen
mask = values != 127  # Máscara booleana para ignorar 127
Z_flat, X_flat, Y_flat, values = Z_flat[mask], X_flat[mask], Y_flat[mask], values[mask]

# Scatter plot 3D con los valores filtrados
ax.scatter(Z_flat, X_flat, Y_flat, c=values, cmap="gray", alpha=0.3, marker=".")

ax.set_xlabel("Profundidad (Z)")
ax.set_ylabel("Ancho (X)")
ax.set_zlabel("Altura (Y)")
ax.set_title("Visualización 3D del Volumen Médico (Ignorando valores 127)")

plt.show()