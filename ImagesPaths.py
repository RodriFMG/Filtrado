import os
import numpy as np


# Si tienen dudas lean los comentarios, es una explicación paso por paso ;b.

# Función que retorna una lista de python con las rutas de TODAS las imagenes del OASIS1.
# Cada posición de la lista serían las imagenes .gif de cada directorio que hay al descargar el OASIS1.

def NumpyListImagesPath(path):
    # Métodos utilizados...
    # os.listdir, en base a una ruta, retorna todos los archivos o directorios que se encuentra
    # en la ruta colocada. (tanto archivos <- files como directorios <- dir). Retorna el NOMBRE
    # de los archivos que hay.

    # os.path.isdir(string), en base al string colocado, examina si es un directorio o un archivo, en caso
    # sea directorio retorna True y si no, retorna False, el string que recibe como parámetro DEBE
    # CONTENER LA RUTA EXACTA DEL ARCHIVO QUE SE QUIERA LEER...

    # Por ello, se realiza un os.path.join(string1, string2), el cual el base al path colocado y la lista
    # de archivos de ese path, concatena ambos string y forman cada ruta de cada archivo presente en ese path.
    directories = [NameDic for NameDic in os.listdir(path) if os.path.isdir(os.path.join(path, NameDic))]

    # Esto se puede cambiar al FSL_SEG, PROCESSED o RAW, como las imagenes despues de cada escaneos estan en
    # RAW (que en si son las que nos interesan) pues utilizaremos este directorio.
    ImageOfDirectory = "/RAW/"

    # Np <- Numpy Array, aprovechando el BroadCast de Numpy, se puede generar un array numpy con las rutas
    # exactas de las imagenes de cada directorio del OASIS.

    # Si no le sale las rutas esperadas es porque definio mal el path-
    NpPathImagesByDirecories = path + "/" + np.array(directories) + ImageOfDirectory

    ImagesByDirectories = []

    # enumerate(list iterable or generator) <- en base a una lista o generador itera la secuencia del objeto
    # para ser utilizado en el valor a utilizar en esa iteración. Este valor será una tupla (index, value)
    # donde index es la posición del value y value es el Object[index].
    for _, ImagesPath in enumerate(NpPathImagesByDirecories):
        # Directamente colocamos la ruta de cada imagen de cada directorio.
        ImagesByDirectory = [ImagesPath + Images for Images in
                             os.listdir(ImagesPath) if Images.endswith('.gif')]

        # Para mantener el orden, cada sublista será TODAS LAS IMAGENES de cada directorio.
        ImagesByDirectories.append(ImagesByDirectory)

    # Se eliminan las imagenes de los directorios en los que tengan menos de 3 scaneos.
    # Esto se debe a que tanto los tensores y listas numpy esperan que TODAS las sublistas
    # tengan las mismas dimensiones, por tanto, como en OASIS1 los directorios se dividen entre
    # 3 y 4 scaneos, donde son pocos los casos donde hay 3, pues elimino para trabajar y manipular
    # los datos ;b

    # Se podría añadir rutas identicas del mismo directorio pero podría causar confusión al distinguir
    # cual es igual o diferente ;b, o se podría crear un archivo con una etiqueta que diferencie
    # las copias entre las demás, pero trabajar con la creación de archivos me da miedo ;b

    for ImagesDic in ImagesByDirectories:
        if len(ImagesDic) <= 3:
            ImagesByDirectories.remove(ImagesDic)

    #print(len(ImagesByDirectories))


    return np.array(ImagesByDirectories)

