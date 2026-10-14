import os

import config
import database
import indexador_service


def ejecutar_indexacion():
    print(" Inicializando Base de Datos Local...")
    database.inicializar_db()

    archivos_ya_listos = database.obtener_archivos_indexados()
    print("Buscando PDFs nuevos en la carpeta...")

    hubo_cambios = False
    for archivo in os.listdir(config.CARPETA_PDFS):
        if (
            archivo.lower().endswith((".pdf", ".docx", ".txt", ".md"))
            and archivo not in archivos_ya_listos
        ):
            hubo_cambios = True
            ruta_completa = os.path.join(config.CARPETA_PDFS, archivo)
            print(f"Procesando: {archivo}...")

            exito, resultado = indexador_service.procesar_un_archivo(ruta_completa, archivo)
            if exito:
                print(f" Guardado en disco ({resultado} páginas).")
            else:
                print(f" Error al procesar: {resultado}")

    if not hubo_cambios:
        print("No se encontraron archivos PDFs nuevos para procesar.")
    else:
        print("\n¡Indexación terminada! Base de datos actualizada con éxito.")


if __name__ == "__main__":
    ejecutar_indexacion()
