import numpy as np
import torch
from ImagesPaths import NumpyListImagesPath
from DicImagesToTensor import DicImagesToTensor


def ImagesPathToTensor(path):
    # Si la función no retorna una ruta esperada o una lista vacio es porque se declaro mal el path.
    ListImagesPath = NumpyListImagesPath(path)
    # print(ListImagesPath)

    PythonListImagesMatrix = [DicImagesToTensor(DicImagesPaths) for DicImagesPaths in ListImagesPath]

    # Se convierte a np.array porque sino es extremadamente mento el proceso.
    TensorListImagesMatrix = torch.tensor(np.array(PythonListImagesMatrix))

    return TensorListImagesMatrix
