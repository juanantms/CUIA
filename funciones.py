import cv2
import numpy as np 
import requests
import os
from deepface import DeepFace
import mediapipe as mp
import face_recognition
from gtts import gTTS
import tempfile
import playsound
import speech_recognition as sr
import glob
import folium
from folium.plugins import MarkerCluster
import webbrowser
import random


api_key = 'p9BPfXKfljgCUZ5MlZ3v9r8HQCKJkXjp'


# Reproducir cadena de caracteres
def ReproducirVoz(texto):
    # Crear un objeto gTTS con el texto proporcionado
    tts = gTTS(text=texto, lang='es')

    # Crear un archivo temporal para almacenar el audio
    archivo_temporal = tempfile.NamedTemporaryFile(delete=False)
    ruta_audio = archivo_temporal.name + ".mp3"

    # Guardar el audio en el archivo temporal
    tts.save(ruta_audio)

    # Reproducir el audio utilizando el reproductor de sistema
    playsound.playsound(ruta_audio)

    # Eliminar el archivo temporal
    os.remove(ruta_audio)

# Reconocimiento de voz
def ReconocerVoz():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("Di algo...")
        audio = r.listen(source)

    texto = ""
    try:
        texto = r.recognize_google(audio, language="es-ES")
        print("Texto reconocido:", texto)
    except sr.UnknownValueError:
        print("No se pudo reconocer el audio")
    except sr.RequestError as e:
        print("Error en la solicitud del reconocimiento de voz:", e)

    return texto

# Reconocimiento Facial
def ReconocimientoFacial():
    # Cargar el clasificador pre-entrenado para la detección de caras
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Inicializar la cámara
    cap = cv2.VideoCapture(0)

    while True:
        # Capturar el cuadro actual de la cámara
        ret, frame = cap.read()

        # Convertir el cuadro a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar caras en la imagen
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Dibujar un rectángulo alrededor de cada cara detectada
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Mostrar la imagen resultante
        cv2.imshow('Reconocimiento Facial', frame)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()

# Reconocimiento Facial + Estimacion de edad
def ReconocimientoFacialEdad():

    detec_cara = mp.solutions.face_detection
    caras = detec_cara.FaceDetection(min_detection_confidence=1, model_selection=2)

    # Inicializar la cámara
    cap = cv2.VideoCapture(0)

    veces = 0
    edades = []
    edad = 0
    mediaEdad = 0

    while True:
        # Leemos fotogramas
        ret, frame = cap.read()
        if ret == False: break
        #frame = imutils.resize(frame, width=500)
        
        veces = veces + 1
        
        # Corregimos el color
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesamos
        rescaras = caras.process(rgb)
        
        if rescaras.detections is not None:
            for cara in rescaras.detections:
                # Ubicacion de la cara
                al, an, c = frame.shape
                box = cara.location_data.relative_bounding_box
                xi, yi, w, h = int(box.xmin*an), int(box.ymin*al), int(box.width*an), int(box.height*al)
                xf, yf = xi + w, yi + h

                # Dibujamos
                cv2.rectangle(frame, (xi,yi), (xf, yf), (255,255,0), 1)

                # Informacion
                if veces == 1 or veces%20 == 0:
                    info = DeepFace.analyze(rgb, actions=['age'], enforce_detection=False)
                    
                    edad = info[0]['age']
                    edades.append(int(edad))
                
                cv2.putText(frame,'Edad: ' + str(edad), (65, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)


        cv2.imshow("Reconocimiento Facial", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    for i in range(2): 
        edades.pop(0)

    mediaEdad = sum(edades) / len(edades)
    mediaEdad = round(mediaEdad)
    print(mediaEdad)

    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()
    
    return mediaEdad

# Comprobar Registro Facial
def ComprobarRegistro():
     
    resultado = False
    nombre = ''

    # Cargar las imágenes almacenadas en el programa
    imagenes_almacenadas = []
    nombres_clientes = []
    face_image_encodings = []

    for nombre_archivo in os.listdir('registrados/'):
        ruta = os.path.join('registrados/', nombre_archivo)
        print(ruta)
        imagen = cv2.imread(ruta)
        #imagen = cv2.resize(imagen, (224, 224))
        #imagen = preprocess_input(imagen)
        if imagen is not None:
            face_loc = face_recognition.face_locations(imagen)[0]
            face_image_encodings.append(face_recognition.face_encodings(imagen, known_face_locations=[face_loc])[0])
            imagenes_almacenadas.append(imagen)
            nombres_clientes.append(nombre_archivo[:-5])


    # Inicializar la cámara
    cap = cv2.VideoCapture(0)


    while True:
        # Leemos fotogramas
        ret, frame = cap.read()
        if ret == False: break
        frame = cv2.flip(frame, 1)
        
        face_locations = face_recognition.face_locations(frame)
        if face_locations != []:
            for face_location in face_locations:
                    face_frame_encodings = face_recognition.face_encodings(frame, known_face_locations=[face_location])[0]
                    for i in range(len(imagenes_almacenadas)):
                        result = face_recognition.compare_faces([face_frame_encodings], face_image_encodings[i])

                        if result[0] == True:
                            resultado = True
                            text = nombres_clientes[i]
                            nombre = text
                            color = (125, 220, 0)
                            break
                        else:
                            text = "Desconocido"
                            color = (50, 50,255)

                    cv2.rectangle(frame, (face_location[3], face_location[2]), (face_location[1], face_location[2]+30), color, -1)
                    cv2.rectangle(frame, (face_location[3], face_location[0]), (face_location[1], face_location[2]), color, 2)
                    cv2.putText(frame, text, (face_location[3], face_location[2]+20), 2, 0.7, (255,255,255), 1)


        cv2.imshow("Comprobacion Registro Facial", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()

    return resultado, nombre

# Registro con Reconocimiento Facial
def RegistroFacial(nombre):
    
    detec_cara = mp.solutions.face_detection
    caras = detec_cara.FaceDetection(min_detection_confidence=1, model_selection=2)

    # Inicializar la cámara
    cap = cv2.VideoCapture(0)

    veces = 0
    edades = []
    edad = 0
    mediaEdad = 0
    ruta_foto = ''

    captura = False

    while True:
        # Leemos fotogramas
        ret, frame = cap.read()
        if ret == False: break
        #frame = imutils.resize(frame, width=500)
        
        veces = veces + 1
        
        # Corregimos el color
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesamos
        rescaras = caras.process(rgb)
        
        if rescaras.detections is not None:
            for cara in rescaras.detections:
                # Ubicacion de la cara
                al, an, c = frame.shape
                box = cara.location_data.relative_bounding_box
                xi, yi, w, h = int(box.xmin*an), int(box.ymin*al), int(box.width*an), int(box.height*al)
                xf, yf = xi + w, yi + h

                # Dibujamos
                cv2.rectangle(frame, (xi,yi), (xf, yf), (255,255,0), 1)

                # Informacion
                if veces == 1 or veces%20 == 0:
                    info = DeepFace.analyze(rgb, actions=['age'], enforce_detection=False)
                    
                    if captura == False and veces>20:
                        ruta_foto = 'registrados/' + nombre + '.jpeg'
                        cv2.imwrite(ruta_foto, frame)

                    edad = info[0]['age']
                    edades.append(int(edad))
                
                cv2.putText(frame,'Edad: ' + str(edad), (65, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)


        cv2.imshow("Reconocimiento Facial", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    for i in range(2): 
        edades.pop(0)

    mediaEdad = sum(edades) / len(edades)
    mediaEdad = round(mediaEdad)
    print(mediaEdad)

    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()
    
    return mediaEdad, ruta_foto

# Reconocimiento de Monumentos
def ReconocimientoMonumento():

    nombre_imagen_referencia = ""

    # Crear el objeto SIFT
    sift = cv2.SIFT_create()
    bf_matcher = cv2.BFMatcher()

    # Obtener la lista de archivos .jpeg en la carpeta "img/"
    lista_archivos = glob.glob("img/*.jpeg")

    # Leer las imágenes de referencia y calcular sus descriptores
    imagenes_referencia = []
    descriptores_referencia = []
    for archivo in lista_archivos:
        imagen_referencia = cv2.imread(archivo)
        gris_referencia = cv2.cvtColor(imagen_referencia, cv2.COLOR_BGR2GRAY)
        kp_referencia, des_referencia = sift.detectAndCompute(gris_referencia, None)
        imagenes_referencia.append(imagen_referencia)
        descriptores_referencia.append(des_referencia)

    # Inicializar la cámara
    cap = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    threshold = 0.01

    while True:
        # Leer el siguiente frame de la cámara
        ret, frame = cap.read()

        # Convertir el frame a escala de grises
        gris_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar keypoints y calcular descriptores del frame actual
        kp_frame, des_frame = sift.detectAndCompute(gris_frame, None)

        max_coincidencias = 0
        indice_coincidente = None
        des_coincidente = None

        if des_frame is not None:
            # Realizar la comparación entre descriptores con todas las imágenes de referencia
            for i in range(len(imagenes_referencia)):
                des_referencia = descriptores_referencia[i]
                if des_referencia is not None:
                    des_referencia = des_referencia.astype(des_frame.dtype)  # Asegurar el mismo tipo de datos
                    coincidencias = bf_matcher.knnMatch(des_referencia, des_frame, k=2)
                    good = []
                    for m, n in coincidencias:
                        if m.distance < 0.7 * n.distance:
                            good.append(m)

                    # Obtener la máxima cantidad de coincidencias
                    if len(good) > max_coincidencias:
                        max_coincidencias = len(good)
                        indice_coincidente = i
                        des_coincidente = des_referencia

        # Dibujar los keypoints y las coincidencias en el frame
        if des_coincidente is not None:
            imagen_resultado = cv2.drawMatchesKnn(imagenes_referencia[indice_coincidente], kp_referencia, frame, kp_frame, [good[:20]], None, flags=cv2.DrawMatchesFlags_DEFAULT)

        # Calcular el porcentaje de coincidencia
        if des_coincidente is not None:
            coincidencia_pct = max_coincidencias / len(des_coincidente)
        else:
            coincidencia_pct = 0

        # Mostrar el nombre de la imagen de referencia si se supera el umbral de coincidencia
        if coincidencia_pct > threshold:
            nombre_imagen_referencia = lista_archivos[indice_coincidente].split("/")[-1].split(".")[0]
            cv2.putText(frame, nombre_imagen_referencia, (65, 50), font, 1, (0, 255, 0), 2)

        # Mostrar el frame con las coincidencias
        cv2.imshow("Reconocimiento de Imagen", frame)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()

    return nombre_imagen_referencia

# Superpone un mapa de granada sobre un marcador de Aruco
def ObtenerMapa():
    # Cargamos el diccionario de aruco
    diccionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

    # Inicializamos parámetros del detector
    parametros = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(diccionario, parametros)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        # Detectamos los marcadores en la imagen
        esquinas, ids, candidatos_malos = detector.detectMarkers(gray)

        if np.all(ids != None):
            aruco = cv2.aruco.drawDetectedMarkers(frame, esquinas)

            # Extraemos los puntos de las esquinas en coordenadas separadas
            c1 = (esquinas[0][0][0][0], esquinas[0][0][0][1])
            c2 = (esquinas[0][0][1][0], esquinas[0][0][1][1])
            c3 = (esquinas[0][0][2][0], esquinas[0][0][2][1])
            c4 = (esquinas[0][0][3][0], esquinas[0][0][3][1])

            copy = frame
            # Leemos la imagen que vamos a sobreponer
            imagen = cv2.imread("img_aruco/mapa.jpg")
            # Extraemos el tamaño de la imagen
            tamaño = imagen.shape
            #tamaño = (514/2, 735/2, 3)
            
            # Organizamos las coordenadas del aruco en una matriz
            puntos_aruco = np.array([c1, c2, c3, c4])

            # Organizamos las coordenadas de la imagen en otra matriz
            puntos_imagen = np.array([
                [0, 0], 
                [tamaño[1]-1, 0], 
                [tamaño[1]-1, tamaño[0]-1], 
                [0, tamaño[0]-1]
            ], dtype=float)

            # Realizamos la superposición de la imagen
            h, estado = cv2.findHomography(puntos_imagen, puntos_aruco)

            # Realizamos la transformación de perspectiva
            perspectiva = cv2.warpPerspective(imagen, h, (copy.shape[1], copy.shape[0]))
            cv2.fillConvexPoly(copy, puntos_aruco.astype(int), 0, 16)
            copy = copy + perspectiva
            cv2.imshow("Mapa de Granada", copy)

        else:
            cv2.imshow("Mapa de Granada", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()

# Obtiene las coordenadas de la direccion pasada por parametro
def ObtenerCoordenadas(direccion):
    # Aquí puedes utilizar una API de geocodificación para obtener las coordenadas de la dirección
    # Implementa esta función según la API que estés utilizando o cómo deseas obtener las coordenadas

    url = f'http://www.mapquestapi.com/geocoding/v1/address?key={api_key}&location={direccion}'

    response = requests.get(url)
    data = response.json()

    if 'results' in data and len(data['results']) > 0:
        result = data['results'][0]
        latitud = result['locations'][0]['latLng']['lat']
        longitud = result['locations'][0]['latLng']['lng']

        return latitud, longitud
    else:
        print('No se encontraron resultados de geocodificación.')
        return None

# Crea una ruta en un mapa desde la direccion origen hasta la direccion destino
def CrearRuta(origen, destino):

    latOri, longOri = ObtenerCoordenadas(origen)
    latDest, longDest = ObtenerCoordenadas(destino)

    url = f'http://www.mapquestapi.com/directions/v2/route?key={api_key}'
    params = {
        'from': f'{latOri},{longOri}',
        'to': f'{latDest},{longDest}',
        'routeType': f'pedestrian',
    }

    response = requests.get(url, params=params)
    data = response.json()
    ruta = data['route']

    coordenadas_ruta = []

    for leg in ruta["legs"]:
        maneuvers = leg['maneuvers']
        for maneuver in maneuvers:
            startPoint = maneuver['startPoint']
            coordenadas_ruta.append((startPoint['lat'], startPoint['lng']))

    return coordenadas_ruta

# Funcion auxiliar. Une las rutas 1 y 2 y necesita las coordenadas de los puntos y sus respectivos nombres
def UnirRutas(nombre_puntos, puntos, ruta1, ruta2):

    mapa = folium.Map(location=[37.17875, -3.59996], zoom_start=15)
    marker_cluster = MarkerCluster().add_to(mapa)

    folium.PolyLine(locations=ruta1, color='blue', weight=3, opacity=1).add_to(mapa)
    folium.PolyLine(locations=ruta2, color='blue', weight=3, opacity=1).add_to(mapa)

    for i in range(len(nombre_puntos)):
        if i == 0:
            folium.Marker([puntos[i][0], puntos[i][1]], icon=folium.Icon(color='green'), popup=nombre_puntos[i]).add_to(marker_cluster)
        elif i == len(nombre_puntos)-1:
            folium.Marker([puntos[i][0], puntos[i][1]], icon=folium.Icon(color='red'), popup=nombre_puntos[i]).add_to(marker_cluster)
        else:   
            folium.Marker([puntos[i][0], puntos[i][1]], icon=folium.Icon(color='blue'), popup=nombre_puntos[i]).add_to(marker_cluster)

    mapa.save('mapa_ruta.html')

    webbrowser.open('mapa_ruta.html')

# Dadas una serie de direcciones de ubicaciones reales, construye una ruta entre ellas
def PlanificarRutaCompleta(ubicaciones):
    
    rutas = []
    ruta1 = []
    nombre_puntos = []
    puntos = []
    
    for i in range(len(ubicaciones)-1):
        origen = ubicaciones[i] + ', Granada, España'
        destino = ubicaciones[i+1] + ', Granada, España'

        r = CrearRuta(origen, destino)

        if i == 0:
            nombre_puntos.append(origen)
            puntos.append(r[0])
        
        if i%2 != 0:
            nombre_puntos.append(origen)
            puntos.append(r[0])
            nombre_puntos.append(destino)
            puntos.append(r[-1])

        if (len(ubicaciones)-1)%2 == 1 and i == len(ubicaciones)-2:
            nombre_puntos.append(destino)
            puntos.append(r[-1])

        rutas.append(r)

    for ruta in rutas[:-1]:
        ruta1.extend(ruta)

    UnirRutas(nombre_puntos, puntos, ruta1, rutas[-1])
        
# Devuelve informacion sobre el monumento identificado, solo si monumento no es ""
def ObtenerInformacion(monumento):

    informacion = ""

    if monumento == "Catedral":
        informacion = "La Santa y Apostólica Iglesia Catedral Metropolitana Basílica de la Encarnación de Granada o más breve S.A.I. Catedral Metropolitana Basílica de Granada es un templo católico de la ciudad española de Granada, sede de la archidiócesis de la ciudad. El templo es una de las obras cumbres del Renacimiento español. Está dedicada al misterio de la Encarnación del Señor. Como muchas otras catedrales de Andalucía, fue construida sobre la parte superior de la mezquita principal de la ciudad." 

    elif monumento == "Plaza_de_Toros":
        informacion = """La Plaza de Toros de Granada, también conocida como Monumental de Frascuelo, es un inmueble histórico de la ciudad española de Granada, donde se celebran corridas de toros y también otros espectáculos públicos. El edificio fue construido entre 1927 y 1928 por el arquitecto local Ángel Casas, a iniciativa de la empresa Sociedad Nueva Plaza de Toros, y está considerado como uno de los emblemas históricos de la arquitectura historicista de Granada.
El edificio es de estilo neomudéjar y desde 1991 está catalogado como Bien de Interés Cultural, tanto por el Ministerio de Cultura como por el gobierno autonómico. Además, la Plaza de toros de Granada es una de los diez cosos taurinos más grandes de España, con un ruedo de 50,40 metros de diámetro y que cuenta con 12 000 localidades, aunque originalmente dispuso de una capacidad de aforamiento de 14 507 personas. Pese a sus grandes dimensiones, desde hace décadas no suele albergar una gran afluencia, llenándose por completo algunas tardes en la Feria del Corpus que se celebra cada mes de junio.
El coso, situado en la Avenida del Doctor Olóriz, fue inaugurado el 30 de septiembre de 1928, con una corrida de toros de la ganadería de Concha y Sierra, en la que actuaron los diestros Manuel Jiménez «Chicuelo», Joaquín Rodríguez «Cagancho» y Fermín Espinosa «Armillita Chico».
En Granada, como en otras ciudades del sur de España existe una tradición taurina destacable. Actualmente los bajos de la plaza de toros alberga diversos bares, restaurantes y pubs donde se realiza la principal actividad comercial del edificio. Esos establecimientos y los que hay en los alrededores constituyen una importante zona de ocio y de tapeo de la ciudad: Plaza Toros."""

    elif monumento == "Alhambra":
        informacion = """La Alhambra es un complejo monumental sobre una ciudad palatina andalusí situada en Granada, España. Consiste en un conjunto de antiguos palacios, jardines y fortalezas (alcazaba, del árabe: القصبة al-qaṣbah ‘ciudadela’) inicialmente concebido para alojar al emir y la corte del reino Nazarí, más tarde como residencia de los reyes de Castilla y de sus representantes. Su singularidad artística radica en los interiores de los palacios nazaríes, cuya decoración está entre las cumbres del arte andalusí, así como en su localización y adaptación, que generó un paisaje nuevo pero totalmente integrado con la naturaleza preexistente. Además, alberga un museo de arte andalusí, la pinacoteca principal de la ciudad y un antiguo convento convertido en parador nacional.
El complejo es gestionado por la institución Patronato de la Alhambra y el Generalife. En 2016 fue el segundo espacio en número de visitantes de España, tras la Sagrada Familia de Barcelona, con 2 615 188 visitantes.
En 1984 fue declarado Patrimonio de la Humanidad por la Unesco, junto con el Generalife y el barrio del Albaicín. """

    elif monumento == "Puerta_de_la_Justicia":
        informacion = """La Puerta de la Justicia es la puerta más monumental de las cuatro puertas exteriores de la Alhambra. Fue construida en 1348 y está junto a un baluarte circular de artillería cristiana. En la Explanada, hay un pilar dedicado a Washington Irving y otro pilar dedicado a Carlos V. La Puerta de la Justicia es conocida también como Puerta de la Explanada y es uno de los símbolos de la Alhambra. Tiene una rica simbología que combina elementos islámicos y góticos. En su interior, cuenta con bóvedas de aristas y una cúpula con decoración en ladrillo rojo. En la salida, se encuentra un retablo construido en 1588. Frente a la puerta, hay una calle antemuro y más adelante se encuentra la Puerta del Vino, punto de inicio del itinerario oficial de visita."""

    else:
        informacion = "Lo siento, no identifico ningun monumento"

    return informacion

# EN funcion de la edad del cliente se elige una ruta
def ElegirRuta(edad):

    ubicaciones = []

    if edad <=35:
        n = random.randint(1,2)
        print (n)
        if n == 1:
            ubicaciones.append("Plaza Nueva")
            ubicaciones.append("Mirador placeta de carvajales")
            ubicaciones.append("calle carrera del darro")
            ubicaciones.append("Cuesta del chapiz")
            ubicaciones.append("mirador de san nicolas")
            ubicaciones.append("mirador de san cristobal")
            ubicaciones.append("jardines de triunfo")
            ubicaciones.append("calle gonzalo gallas")
        else:
            ubicaciones.append("plaza nueva")
            ubicaciones.append("Puerta de la justicia")
            ubicaciones.append("Paseo del Generalife")
            ubicaciones.append("Cuesta del rey chico")
            ubicaciones.append("Cuesta del chapiz")
            ubicaciones.append("La chumbera")
            ubicaciones.append("mirador de san nicolas")
            ubicaciones.append("plaza nueva")

    elif edad > 35 and edad <=55:
        n = random.randint(1,2)
        if n == 1:
            ubicaciones.append("Fuente de las batallas")
            ubicaciones.append("plaza de Bib-rambla")
            ubicaciones.append("plaza de las pasiegas")
            ubicaciones.append("calle oficios")
            ubicaciones.append("plaza nueva")
            ubicaciones.append("calle carrera del darro")
            ubicaciones.append("Cuesta del chapiz")
            ubicaciones.append("Mirador de san nicolas")
            ubicaciones.append("Calle Reyes Catolicos")
            ubicaciones.append("Plaza del carmen")
            ubicaciones.append("Calle carrera de la virgen")
        else:
            ubicaciones.append("plaza nueva")
            ubicaciones.append("Puerta de la justicia")
            ubicaciones.append("Paseo del Generalife")
            ubicaciones.append("Cuesta del rey chico")
            ubicaciones.append("Cuesta del chapiz")
            ubicaciones.append("La chumbera")
            ubicaciones.append("mirador de san nicolas")
            ubicaciones.append("plaza nueva")
    elif edad > 55: 
        ubicaciones.append("fuente de las batallas")
        ubicaciones.append("Puerta de Bib-Rambla")
        ubicaciones.append("Plaza de las pasiegas")
        ubicaciones.append("Barrio de san jeronimo")
        ubicaciones.append("jardines de triunfo")
        ubicaciones.append("plaza albert einstein")
        ubicaciones.append("recogidas")


    return ubicaciones
