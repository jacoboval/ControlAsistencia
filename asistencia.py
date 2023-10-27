import cv2
import face_recognition as fr
import os
import numpy
from datetime import  datetime


# crear base de datos
ruta = 'Empleados'
mis_imagenes = []
nombres_empleados = []
lista_empleados = os.listdir(ruta)

#print(lista_empleados)
for nombre in lista_empleados:
    imagen_actual = cv2.imread(f'{ruta}\{nombre}')
    mis_imagenes.append(imagen_actual)
    nombres_empleados.append(os.path.splitext(nombre)[0])

print(nombres_empleados)

# codificar imagenes
def codificar(imagenes):

    # crear una lista nueva
    lista_codificada = []

    # pasar todas las imagenes a gjpg
    for imagen in imagenes:
        imagen = cv2.cvtColor(imagen,cv2.COLOR_BGR2RGB)

        # CODIFICAR
        codificado = fr.face_encodings(imagen)[0]

        #agrqagar a la lista
        lista_codificada.append(codificado)

        # devolver lista codificada
    return lista_codificada

# registrar lso ingresos
def registrar_ingresos(persona):
    f = open('registros.csv', 'r+')
    lista_datos = f.readlines()
    nombres_registros = []
    for linea in lista_datos:
        ingreso = linea.split(',')
        nombres_registros.append([0])
    if persona not in nombres_registros:
        ahora = datetime.now()
        string_ahora = ahora.strftime('%H:%M:%S')
        string_fecha = ahora.date()
        f.writelines(f'\n{persona},{string_fecha} {string_ahora}')

lista_empleados_codificada = codificar(mis_imagenes)
#print(len(lista_empleados_codificada))

# tomar una omagen de caamre web, 0 es indice
captura = cv2.VideoCapture(0,cv2.CAP_DSHOW)

# leer la img d la camara
exito, imagen = captura.read()

if not exito:
    print("No s ha logrado caprurar la imagen")
else:
    # recomnocer cara en captura
    cara_captura = fr.face_locations(imagen)

    # coddifuicar carqa cpturada
    cara_captura_codificada = fr.face_encodings(imagen,cara_captura)

    # buscar coincidencias

    for caracodif, caraubic in zip(cara_captura_codificada,cara_captura):
        coincidencias = fr.compare_faces(lista_empleados_codificada,caracodif) # ubicacion
        distancias = fr.face_distance(lista_empleados_codificada,caracodif)  # distamcia

        print(distancias)

        indice_coincidencia = numpy.argmin(distancias)

        # mostrar coincidencias si las hay
        if distancias[indice_coincidencia] > 0.6:
            print("No coindice con ningun empleado.")
        else:
            #print("Welcome")

            # buscar nombre del empeladop encontrado
            nombre = nombres_empleados[indice_coincidencia]

            y1, x1, y2, x2 = caraubic
            cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.rectangle(imagen, (x1, y2 - 35),(x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(imagen,nombre, (x1 - 180, y2 -6), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            registrar_ingresos(nombre)


            # mostrar imagen
            cv2.imshow("Imagen web",imagen)

            # mantemner ventana abierta
            cv2.waitKey()



