import os
import sys

# Rutas del sistema
# Si corre como ejecutable (PyInstaller), los datos van junto al .exe / binario,
# no en la carpeta temporal donde se descomprime el programa.
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_PDFS = os.path.join(BASE_DIR, "mis_pdfs")
DB_PATH = os.path.join(BASE_DIR, "biblioteca.db")


# Carpeta de modelos ONNX. Si el ejecutable fue empaquetado con el modelo
# bundleado, apunta adentro del bundle; si no, a BASE_DIR/model_cache.
def _modelos_dir():
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        return os.path.join(base, "model_cache")
    return os.path.join(BASE_DIR, "model_cache")


MODELOS_DIR = _modelos_dir()

# Asegurar que las carpetas existan al iniciar
if not os.path.exists(CARPETA_PDFS):
    os.makedirs(CARPETA_PDFS)
    os.makedirs(MODELOS_DIR, exist_ok=True)


def cargar_env():
    """Carga variables de entorno manualmente desde un archivo .env si existe."""
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for linea in f:
                if "=" in linea and not linea.startswith("#"):
                    clave, valor = linea.strip().split("=", 1)
                    os.environ[clave] = valor


# Cargar el entorno por si se requiere en el futuro
cargar_env()
