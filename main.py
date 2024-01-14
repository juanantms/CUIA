import funciones
import cliente as cl
import sys
import numpy as np


estado, nombre = funciones.ComprobarRegistro()


cliente = cl.Cliente()

if estado == False:
    funciones.ReproducirVoz("¿Quiere registrarse?")
    respuesta = funciones.ReconocerVoz()
    if respuesta == "si":
        funciones.ReproducirVoz("Comienzo del registro")
        funciones.ReproducirVoz("Como te llamas")
        nombre = funciones.ReconocerVoz()
        funciones.ReproducirVoz("Hola " + nombre)
        edad, ruta_foto = funciones.RegistroFacial(nombre)
        print(ruta_foto)
        # Crear una instancia de la clase Persona
        cliente = cl.Cliente(nombre, edad, ruta_foto)
        # Guardar los datos de la persona en un archivo
        cliente.guardar_datos("datos/datos"+nombre+".json")
    else:
        funciones.ReproducirVoz("Hasta luego.")
        sys.exit()
else:
    funciones.ReproducirVoz("Bienvenido " + nombre)
    # Cargar los datos de una persona desde el archivo
    cliente = cl.Cliente.cargar_datos("datos/datos"+nombre+".json")


while True:

    respuesta = funciones.ReconocerVoz()

    if respuesta == "ruta":
        funciones.ReproducirVoz("¿Desea que le planifique una ruta por Granada?")
        respuesta = funciones.ReconocerVoz()
        if respuesta == "si":
            funciones.ReproducirVoz("¿Está lloviendo?")
            tiempo = funciones.ReconocerVoz()
            if tiempo == "no":
                funciones.ReproducirVoz("¿En que calle estas de Granada?")
                ubicacion = funciones.ReconocerVoz()
                funciones.ReproducirVoz("Vale, espere un segundo.")
                print(cliente.edad)
                ubi = funciones.ElegirRuta(cliente.edad)
                ubicaciones = []
                ubicaciones.append(ubicacion)
                ubicaciones = np.concatenate((ubicaciones, ubi))
                print(ubicaciones)
                funciones.PlanificarRutaCompleta(ubicaciones)
            elif tiempo == "si": 
                funciones.ReproducirVoz("Lo siento, lloviendo te recomiendo quedarte en casa y esperar a que haga mejor tiempo.")


    if respuesta == "abrir mapa":
        funciones.ObtenerMapa()
                                                                                            
    if respuesta == "abrir cámara":
        monumento = funciones.ReconocimientoMonumento()
        if monumento != "":
            respuesta2 = funciones.ReconocerVoz()
            if respuesta2 == "más información":
                info = funciones.ObtenerInformacion(monumento)
                funciones.ReproducirVoz(info)
    
    if respuesta == "ayuda":
        info = """Los comandos de voz son los siguientes:
        ruta. Si quiere planificar una ruta.
        abrir mapa. Si quiere ver un mapa de granada.
        abrir cámara. Si quiere reconocer y obtener informacion de un monumento de la ciudad a traves de la cara."""
        funciones.ReproducirVoz(info)


    if respuesta == "salir":
        funciones.ReproducirVoz("Gracias, hasta luego.")
        break


