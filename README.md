# lbp-pyramid

## Ejecuciones

### Abrir jupyter-notebook

1. Abrir la terminal en la carpeta /home/fer/Nextcloud/Master-IA/TFM/lbp-pyramid
2. Ejecutar el comando 'source venv/bin/activate'
3. Ejecutar el comando 'jupyter-notebook'

### Generar la BBDD

Fichero create_db.py, ejecución de la función main.

Métodos:
- get_datasets_by_scale: proporciona múltiples BBDD, reescalando tanto la imagen como el gold standard, con la finalidad de aplicar múltiples modelos de clasificación, uno por escala, y ensamblar posteriormente los resultados obtenidos para constituir el clasificador final.
- get_pyramid_dataset: BBDD única en la que se encuentra la información de las diferentes escalas con la finalidad de aplicar un único clasificador sobre el conjunto total de datos.

### Clasificación de los píxeles

Fichero main.py.

Métodos:
- get_datasets_by_scale: proporciona múltiples BBDD, reescalando tanto la imagen como el gold standard, con la finalidad de aplicar múltiples modelos de clasificación, uno por escala, y ensamblar posteriormente los resultados obtenidos para constituir el clasificador final.
- get_pyramid_dataset: BBDD única en la que se encuentra la información de las diferentes escalas con la finalidad de aplicar un único clasificador sobre el conjunto total de datos.


## Preprocesamiento

### Tratamiento inicial de la imagen

- La lectura de las imágenes se realiza en un único canal, es decir, en escala de grises.
- Se aplica la máscara proporcionada junto con las imágenes para anular el valor, y por lo tanto, descartar, aquellos píxeles que carezcan de relevancia.

### Procesamiento inicial de la imagen

1. En primer lugar, se aplica un filtro bilateral que permite eliminar parte del ruido sin disminuir la definición de los bordes existentes en la medida de lo posible.

![alt text](https://raw.githubusercontent.com/f-castellanos/lbp-pyramid/preprocess/readme_media/preprocess_1_mask_noise_reduction.png)

2. En una segunda fase, se normaliza la imagen mediante la normalización de los histogramas en regiones locales de la imagen, permitiendo así resaltar los vasos sanguíneos de manera independiente a la intensidad luminosa de la imagen en la región que se encuentre.

![alt text](https://raw.githubusercontent.com/f-castellanos/lbp-pyramid/preprocess/readme_media/preprocess_2_normalization.png)

3. Se aplica de nuevo un filtro bilateral como última fase de eliminación de ruido.

![alt text](https://raw.githubusercontent.com/f-castellanos/lbp-pyramid/preprocess/readme_media/preprocess_3_2nd_noise_reduction.png)

### Aplicación del operador LBP - Generación de la base de datos

1. Dado que la imagen va a ser reescalada repetidamente a inferiores dimensiones en la simulación de la variación del radio del operador LBP, y dado que el reescalado se realiza reduciendo la dimensión de la imagen a 1/4 de su tamaño previo, es preciso que las dimensiones de altura y ancho de la imagen sean múltiplos de 2 tantas veces como reescalados se lleven a cabo. Con la finalidad de no distorsionarlas ni generar ruido, y dado que gracias a la máscara existe en todas ellas una región externa de píxeles nulos, se lleva a cabo una adición de píxeles de valor 0, aumentando dicha región, para obtener las dimensiones finales deseadas. Es decir, se añaden márgenes de información vacía en las imágenes.
2. Se lleva a cabo el reescalado de la imagen. [get_datasets_by_scale] Se realiza la misma operación para el gold standard.
3. Se aplica a las diferentes escalas el operador LBP (conservando invariante el valor del parámetro radio del operador ya que se simula mediante el reescalado la variacion de su valor).
4. [get_pyramid_dataset] Dado que en la base de datos generada cada píxel constituye una observación y el número de píxeles es variable, se tomará como instancia de la misma cada píxel de la imagen original, de modo que los valores LBP obtenidos a partir de las escalas inferiores serán asignados a todos los píxeles originales a partir de los que se ha generado el píxel de la nueva escala. Es decir, en el primer reescalado la imagen escalada tiene 1/4 del número de píxeles original, con lo que los nuevos píxeles equivalen a cuatro píxeles de la imagen original, los cuales poseerán en la base de datos el valor asignado al nuevo píxel en la operación LBP.

![alt text](https://raw.githubusercontent.com/f-castellanos/lbp-pyramid/preprocess/readme_media/preprocess_4_lbp_1.png)
![alt text](https://raw.githubusercontent.com/f-castellanos/lbp-pyramid/preprocess/readme_media/preprocess_4_lbp_2.png)
![alt text](https://raw.githubusercontent.com/f-castellanos/lbp-pyramid/preprocess/readme_media/preprocess_4_lbp_3.png)
![alt text](https://raw.githubusercontent.com/f-castellanos/lbp-pyramid/preprocess/readme_media/preprocess_4_lbp_4.png)
![alt text](https://raw.githubusercontent.com/f-castellanos/lbp-pyramid/preprocess/readme_media/preprocess_4_lbp_5.png)
![alt text](https://raw.githubusercontent.com/f-castellanos/lbp-pyramid/preprocess/readme_media/preprocess_4_lbp_6.png)

5. [get_pyramid_dataset] Se añade la información de la imagen preprocesada como una característica adicional a los cálculos del operador LBP.
6. Se elimina de la base de datos los píxeles que no forman parte de la máscara.
7. Como procedimiento final de la constitución de la base de datos, se añade una nueva variable correspondiente a la etiqueta de cada píxel de la imagen original, es decir, una valor binario indicativo de la pertenencia a un vaso sanguíneo.

Finalmente, en la matriz de información obtenida, cada fila constituye un píxel de la imagen de mayor resolución y cada columna corresponde a una resolución.

### Procesamiento tras la aplicación del operador

Se aplica una función que realiza la codificación de las variables (one hot encode) para garantizar que los datos provenientes del operador LBP son empleados de manera discreta.
