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

### Procesamiento de la imagen

1. En primer lugar, se aplica un filtro bilateral que permite eliminar parte del ruido sin disminuir la definición de los bordes existenes en la medida de lo posible.
2. En una segunda fase, se normaliza la imagen mediante la normalización de los histogramas en regiones locales de la imagen, permitiendo así resaltar los vasos sanguíneos de manera independiente a la intensidad luminosa de la imagen en la región que se encuentre.
