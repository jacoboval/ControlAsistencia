import cv2
import  face_recognition as fr

#cargar imafgen
foto_control= fr.load_image_file('fotoA.jpg')
foto_prueba = fr.load_image_file('fotoB.jpg')
#foto_prueba2 = fr.load_image_file('fotoC.jpg')

#pasar imagenes a RGB
foto_control = cv2.cvtColor(foto_control,cv2.COLOR_BGR2RGB)
foto_prueba = cv2.cvtColor(foto_prueba,cv2.COLOR_BGR2RGB)
#foto_prueba2 = cv2.cvtColor(foto_prueba2,cv2.COLOR_BGR2RGB)


#localizar caras control
lugar_cara_A = fr.face_locations(foto_control)[0]
cara_codificada_A = fr.face_encodings(foto_control)[0]

#localizar caras control
lugar_cara_B = fr.face_locations(foto_prueba)[0]
cara_codificada_B = fr.face_encodings(foto_prueba)[0]

# #localizar caras control2
# lugar_cara_C = fr.face_locations(foto_prueba2)[0]
# cara_codificada_C = fr.face_encodings(foto_prueba2)[0]


# mostrar rectangulo
cv2.rectangle(foto_control,
              (lugar_cara_A[3],lugar_cara_A[0]),
              (lugar_cara_A[1], lugar_cara_A[2]),
              (0,255,0),
              2
             )

# mostrar rectangulo
cv2.rectangle(foto_prueba,
              (lugar_cara_B[3],lugar_cara_B[0]),
              (lugar_cara_B[1], lugar_cara_B[2]),
              (0,255,0),
              2
             )

# # mostrar rectangulo
# cv2.rectangle(foto_prueba2,
#               (lugar_cara_C[3],lugar_cara_C[0]),
#               (lugar_cara_C[1], lugar_cara_C[2]),
#               (0,255,0),
#               2
#              )

#print(lugar_cara_A)

#realizar comparaciones

resultado = fr.compare_faces([cara_codificada_A],cara_codificada_B)
#print(resultado)

# medida de la distancia
distancia = fr.face_distance([cara_codificada_B],cara_codificada_B)
print(distancia)

# mostrar resultado
cv2.putText(foto_prueba,
            f'{resultado}{distancia.round(2)}',
            (50,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
            )
#mostrar imagenes
cv2.imshow('foto control', foto_control)
cv2.imshow('Foto Prueba', foto_prueba)
#cv2.imshow('Foto Prueba2', foto_prueba2)

#mantener programa abierto
cv2.waitKey(0)
