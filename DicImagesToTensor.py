import numpy as np
from PIL import Image

def DicImagesToTensor(DicImagesPaths):


    DicTensorMatrix = []
    for ImagePath in DicImagesPaths:

        gif = Image.open(ImagePath)
        frame = np.array(gif.convert("RGB"), dtype=np.float32)
        DicTensorMatrix.append(frame)

    return np.array(DicTensorMatrix)
