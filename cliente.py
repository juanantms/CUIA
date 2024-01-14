import json

class Cliente:
    def __init__(self, nombre='', edad=0, foto=''):
        self.nombre = nombre
        self.edad = edad
        self.foto = foto

    def guardar_datos(self, archivo):
        datos = {
            "nombre": self.nombre,
            "edad": self.edad,
            "foto": self.foto
        }
        with open(archivo, 'w') as file:
            json.dump(datos, file)

    @classmethod
    def cargar_datos(cls, archivo):
        with open(archivo, 'r') as file:
            datos = json.load(file)

        cliente = cls()
        cliente.nombre = datos["nombre"]
        cliente.edad = datos["edad"]
        cliente.foto = datos["foto"]
        
        return cliente

