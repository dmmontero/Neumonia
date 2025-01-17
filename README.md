# Hola! Bienvenido a la herramienta para la detección rápida de neumonía

Deep Learning aplicado en el procesamiento de imágenes radiográficas de tórax en formato DICOM con el fin de clasificarlas en 3 categorías diferentes:

1. Neumonía Bacteriana (**bacterina**)

2. Neumonía Viral (**normal**)

3. Sin Neumonía (**viral**)

Aplicación de una técnica de explicación llamada Grad-CAM para resaltar con un mapa de calor las regiones relevantes de la imagen de entrada.

---

## Uso de la herramienta

A continuación le explicaremos cómo empezar a utilizarla.

Requerimientos necesarios para el funcionamiento:

- Instale Anaconda para Windows siguiendo las siguientes instrucciones:
  <https://docs.anaconda.com/anaconda/install/windows/>

- Abra Anaconda Prompt y ejecute las siguientes instrucciones:

1. Crea un ambiente virtual con la versión 3.11 de Python\
   _conda create -n neumonia python=3.11_

2. Activar el ambiente virtual creado:\
   _conda activate neumonia_

3. Ir a la carpeta del proyecto:\
   _cd Neumonia_

4. Instalar los paquetes definidos para el proyecto:\
   _pip install -r requirements.txt_

5. Ejecutar el proyecto:\
   _python detector_neumonia.py_

Uso de la Interfaz Gráfica:

- Ingrese la cédula del paciente en la caja de texto (opcional)
- Presione el botón 'Cargar Imagen', seleccione la imagen del explorador de archivos del computador (Imagenes de prueba en <https://drive.google.com/drive/folders/1WOuL0wdVC6aojy8IfssHcqZ4Up14dy0g?usp=drive_link>)
- Presione el botón 'Predecir' y espere unos segundos hasta que observe los resultados
- Presione el botón 'Guardar' para almacenar la información del paciente en un archivo excel con extensión .csv
- Presione el botón 'PDF' para descargar un archivo PDF con la información desplegada en la interfaz
- Presión el botón 'Borrar' si desea cargar una nueva imagen

---

## Imágen Docker

El proyecto puede se ejecutado desde una Imagen Docker, para ello debe ejecutar los siguientes pasos:

1. Crear la imagen Docker:
   Una vez se encuentre en la raiiz del proyecto ejecuta el siguiente comando para crear la imágen

   **docker build -t neumonia_detector .**

2. Ejecutar la imagen Docker:
   Debe tener en cuenta que el sistema debe contar con un servidor XWindows activo en en nuestro caso ejecutamos
   la prueba en ambiente Windows e instalamos Xming el cual puede descargarse en: <https://sourceforge.net/projects/xming/>

   **docker run -it -e DISPLAY="host.docker.internal:0.0" -v "c:\DICOM\:/downloads" neumonia_detector**

   Tener en cuenta el parámetro **-v** ( de volumen) el cual nos permite cargar archivos de sistema actual (c:\DICOM) en el sistema de archivos de la imágen (carpeta download en este caso), el parametro -e que nos permite exportar el display.

   Debe por tanto mapear la carpeta que contenga las imágenes de prueba para cargarlas al sistema de archivo de la imágen.

---

## Arquitectura de archivos propuesta

Los archivos se encuentra dentro de la carpeta **src** a excepcion del _detector_neumonia.py_
que es el punto de entra de la app y se encuenta en la raíz del proyecto.

## detector_neumonia.py

Contiene el diseño de la interfaz gráfica utilizando Tkinter.

Los botones llaman métodos contenidos en otros scripts.

## integrator.py

Es un módulo que integra los demás scripts y retorna solamente lo necesario para ser visualizado en la interfaz gráfica.
Retorna la clase, la probabilidad y una imagen el mapa de calor generado por Grad-CAM.

## read_img.py

Script que lee la imagen en formato DICOM para visualizarla en la interfaz gráfica. Además, la convierte a arreglo para su preprocesamiento.

## preprocess_img.py

Script que recibe el arreglo proveniento de read_img.py, realiza las siguientes modificaciones:

- resize a 512x512
- conversión a escala de grises
- ecualización del histograma con CLAHE
- normalización de la imagen entre 0 y 1
- conversión del arreglo de imagen a formato de batch (tensor)

## load_model.py

Script que lee el archivo binario del modelo de red neuronal convolucional previamente entrenado llamado **'conv_MLP_84.h5.h5'**.
el cual debe cargarse a la carpeta modelos y se puede descaragar aqui:

**<https://drive.google.com/file/d/18rgX66x7eMHci0bAimoycCe8BQLjBWK_/view?usp=sharing>**

## grad_cam.py

Script que recibe la imagen y la procesa, carga el modelo, obtiene la predicción y la capa convolucional de interés para obtener las características relevantes de la imagen.

---

## Acerca del Modelo

La red neuronal convolucional implementada (CNN) es basada en el modelo implementado por F. Pasa, V.Golkov, F. Pfeifer, D. Cremers & D. Pfeifer
en su artículo Efcient Deep Network Architectures for Fast Chest X-Ray Tuberculosis Screening and Visualization.

Está compuesta por 5 bloques convolucionales, cada uno contiene 3 convoluciones; dos secuenciales y una conexión 'skip' que evita el desvanecimiento del gradiente a medida que se avanza en profundidad.
Con 16, 32, 48, 64 y 80 filtros de 3x3 para cada bloque respectivamente.

Después de cada bloque convolucional se encuentra una capa de max pooling y después de la última una capa de Average Pooling seguida por tres capas fully-connected (Dense) de 1024, 1024 y 3 neuronas respectivamente.

Para regularizar el modelo utilizamos 3 capas de Dropout al 20%; dos en los bloques 4 y 5 conv y otra después de la 1ra capa Dense.

El modelo puede descargarse aquí:

**<https://drive.google.com/file/d/18rgX66x7eMHci0bAimoycCe8BQLjBWK_/view?usp=sharing>**

## Acerca de Grad-CAM

Es una técnica utilizada para resaltar las regiones de una imagen que son importantes para la clasificación. Un mapeo de activaciones de clase para una categoría en particular indica las regiones de imagen relevantes utilizadas por la CNN para identificar esa categoría.

Grad-CAM realiza el cálculo del gradiente de la salida correspondiente a la clase a visualizar con respecto a las neuronas de una cierta capa de la CNN. Esto permite tener información de la importancia de cada neurona en el proceso de decisión de esa clase en particular. Una vez obtenidos estos pesos, se realiza una combinación lineal entre el mapa de activaciones de la capa y los pesos, de esta manera, se captura la importancia del mapa de activaciones para la clase en particular y se ve reflejado en la imagen de entrada como un mapa de calor con intensidades más altas en aquellas regiones relevantes para la red con las que clasificó la imagen en cierta categoría.

---

## Test Cases

En la carpeta **test** del proyecto se encuentra los casos de prueba creados con **unittest** para ejecutar un caso de prueba específico ejecute el siguiente comando:

**_python -m unittest discover .\test test_load_model.py_**

Donde _test_load_model.py_ en el archivo que contiene la prueba este comando ejectar todas las pruebas creadas.

---

## Proyecto original realizado por

Isabella Torres Revelo - <https://github.com/isa-tr>\
Nicolas Diaz Salazar - <https://github.com/nicolasdiazsalazar>

## Modificado por

**Danny Mauricio Montero - <http://github.com/dmmontero>**
