import numpy as np


# LLeva el valor a un rango [0, 1] según los datos del array.
def LinearNormalization(Value, Min, Max):

    # Para evitar resultados muy pequeños, se coloca en flotantes para que la operación retorne un flotante
    # con alta capacidad de aguantar varios decimales evitando el redondeo
    Min, Max = float(Min), float(Max)

    # Sin esto me salia otro aviso, aunque no debería pasar.
    denominator = 1 if Max == Min else Max - Min

    return (Value - Min) / denominator


def MedicalFormatToPNGAndGenerator(MedicalImg):

    [dimz, _, _] = MedicalImg.shape

    # saca el mínimo de cada matriz YxX, el np.float_num, permite que las operaciones que se hagan con este array
    # tengan alta capacidad de aguantar varios números decimales sin redondear. (lo cual se busca porque
    # el array contiene números muy grandes y muy pequeños en Max y Min respectivamente, por tanto
    # evitaremos el posible redondeo).
    MinForVolumen = np.min(MedicalImg, axis=(1, 2)).astype(np.float64)
    MaxForVolumen = np.max(MedicalImg, axis=(1, 2)).astype(np.float64)

    # Se multiplica por 255, para que este en el rango de pixel value.
    # el uint8, es un tipo de dato que significa:
    # u <- Valores sin signo
    # int8 <- Solo valores en el pixel range [0, 255]

    # Lo retorno como un generador para optimizar el proceso :b

    return [
        (LinearNormalization(MedicalImg[volumen], MinForVolumen[volumen], MaxForVolumen[volumen]) * 255).astype(np.uint8)
            for volumen in range(dimz)]
