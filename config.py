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


def _batch_size_para_ram():
    """Elige BATCH_SIZE según RAM total. Fallback a 32 si no se puede detectar."""
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                for linea in f:
                    if linea.startswith("MemTotal:"):
                        gb = int(linea.split()[1]) * 1024 / 1e9
                        break
                else:
                    return 32
        elif sys.platform == "darwin":
            import subprocess

            gb = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()) / 1e9
        elif sys.platform == "win32":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            s = _MS()
            s.dwLength = ctypes.sizeof(s)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            gb = s.ullTotalPhys / 1e9
        else:
            return 32
    except Exception:
        return 32  # fallback conservador

    if gb < 6:
        return 16
    if gb < 10:
        return 32
    return 64


BATCH_SIZE = _batch_size_para_ram()
print("BATCH_SIZE elegido por RAM: %d" % BATCH_SIZE)

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
