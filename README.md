# lbp-pyramid

## Abrir jupyter-notebook

1. Abrir la terminal en la carpeta /home/fer/Nextcloud/Master-IA/TFM/lbp-pyramid
2. Ejecutar el comando 'source venv/bin/activate'
3. Ejecutar el comando 'jupyter-notebook'

## Preprocesamiento

### Tratamiento inicial de la imagen

- La lectura de las imágenes se realiza en un único canal, es decir, en escala de grises.
- Se aplica la máscara proporcionada junto con las imágenes para anular el valor, y por lo tanto, descartar, aquellos píxeles que carezcan de relevancia.
- La utilización de la imágen a diferentes escalas, conservando la equivalencia entre los píxeles, exige que las dimensiones de la misma sean múltiplos de 2 tantas veces como cambios de escala se realicen, con lo que es precisa una modificacion de la escala original de las imágenes. Con la finalidad de no distorsionarlas ni generar ruido, y dado que gracias a la máscara existe en todas ellas una región externa de píxeles nulos, se lleva a cabo una adición de píxeles de valor 0, aumentando dicha región, para obtener las dimensiones finales deseadas.

### Procesamiento inicial de la imagen

1. En primer lugar, se aplica un filtro bilateral que permite eliminar parte del ruido sin disminuir la definición de los bordes existentes en la medida de lo posible.

![alt text](https://raw.githubusercontent.com/f-castellanos/lbp-pyramid/preprocess/readme_media/preprocess_1_mask_noise_reduction.png)

2. En una segunda fase, se normaliza la imagen mediante la normalización de los histogramas en regiones locales de la imagen, permitiendo así resaltar los vasos sanguíneos de manera independiente a la intensidad luminosa de la imagen en la región que se encuentre.

![alt text](https://raw.githubusercontent.com/f-castellanos/lbp-pyramid/preprocess/readme_media/preprocess_2_normalization.png)

3. Se aplica de nuevo un filtro bilateral como última fase de eliminación de ruido.

![alt text](https://raw.githubusercontent.com/f-castellanos/lbp-pyramid/preprocess/readme_media/preprocess_3_2nd_noise_reduction.png)

### Aplicación del operador LBP
1. Se lleva a cabo el reescalado de la imagen.
2. Se aplica a las diferentes escalas el operador LBP (conservando invariante el valor del parámetro radio del operador ya que se simula mediante el reescalado la variacion de su valor).

# repeat_pixels que hace?

### Procesamiento tras la aplicación del operador

Se aplica una función que realiza la codificación de las variables (one hot encode) para garantizar que los datos provenientes del operador LBP son empleados de manera discreta.
