# -*- coding: utf-8 -*-
"""Monitor de rendimiento: BM25 + e5-small + RRF."""
import argparse
import json
import os
import platform
import random
import shutil
import statistics
import sys
import tempfile
import threading
import time

import numpy as np

import config
import database

for _flujo in (sys.stdout, sys.stderr):
    try:
        _flujo.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


VOCABULARIO = [
    "sistema",
    "proceso",
    "memoria",
    "archivo",
    "datos",
    "red",
    "servidor",
    "cliente",
    "usuario",
    "funcion",
    "variable",
    "objeto",
    "clase",
    "modulo",
    "libreria",
    "algoritmo",
    "estructura",
    "consulta",
    "indice",
    "tabla",
    "registro",
    "campo",
    "valor",
    "cadena",
    "numero",
    "entero",
    "flotante",
    "booleano",
    "lista",
    "conjunto",
    "diccionario",
    "iterador",
    "generador",
    "decorador",
    "contexto",
    "excepcion",
    "error",
    "evento",
    "hilo",
    "cola",
    "pila",
    "arbol",
    "grafo",
    "nodo",
    "arista",
    "camino",
    "busqueda",
    "ordenamiento",
    "complejidad",
    "rendimiento",
    "optimizacion",
    "compilador",
    "interprete",
    "sintaxis",
    "semantica",
    "tipado",
    "polimorfismo",
    "herencia",
    "encapsulamiento",
    "abstraccion",
    "interfaz",
    "protocolo",
]
CONECTORES = [
    "el",
    "la",
    "los",
    "las",
    "un",
    "una",
    "de",
    "en",
    "con",
    "por",
    "para",
    "y",
    "o",
    "que",
    "se",
    "su",
    "del",
    "al",
]
CONSULTAS_BENCH = [
    "algoritmo",
    "memoria del sistema",
    "consulta de datos",
    "estructura de red",
    "busqueda binaria",
    "servidor local",
    "proceso y archivo",
    "palabra inventada zzz",
]


def _generar_corpus(libros, paginas_por_libro, semilla=20251014):
    rng = random.Random(semilla)
    corpus = {}
    for i in range(libros):
        nombre = "libro_%03d.txt" % (i + 1)
        paginas = []
        for _ in range(paginas_por_libro):
            palabras = []
            for _ in range(120):
                palabras.append(rng.choice(VOCABULARIO))
                if rng.random() < 0.4:
                    palabras.append(rng.choice(CONECTORES))
            texto = " ".join(palabras)
            paginas.append(texto[0].upper() + texto[1:] + ".")
        corpus[nombre] = paginas
    return corpus


# -- info del sistema / memoria (igual que antes) ---------------------------


def _nucleos():
    return os.cpu_count() or 1


def _ram_total_bytes():
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo", "r") as f:
                for linea in f:
                    if linea.startswith("MemTotal:"):
                        return int(linea.split()[1]) * 1024
        elif sys.platform == "darwin":
            import subprocess

            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
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
            return int(s.ullTotalPhys)
    except Exception:
        pass
    return None


def _rss_bytes():
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/self/status", "r") as f:
                for linea in f:
                    if linea.startswith("VmRSS:"):
                        return int(linea.split()[1]) * 1024
        elif sys.platform == "darwin":
            import resource

            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        elif sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class _P(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            p = _P()
            p.cb = ctypes.sizeof(p)
            ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(p), p.cb
            )
            return int(p.WorkingSetSize)
    except Exception:
        pass
    return None


def _fmt_bytes(n):
    if n is None:
        return "desconocido"
    v = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if v < 1024.0:
            return "%.1f %s" % (v, u)
        v /= 1024.0
    return "%.1f PB" % v


class _MuestreadorRSS(threading.Thread):
    def __init__(self, intervalo=0.03):
        threading.Thread.__init__(self)
        self.daemon = True
        self.intervalo = intervalo
        self.pico = 0
        self._parar = threading.Event()

    def run(self):
        while not self._parar.is_set():
            r = _rss_bytes()
            if r and r > self.pico:
                self.pico = r
            self._parar.wait(self.intervalo)

    def detener(self):
        self._parar.set()
        self.join(timeout=1.0)


# -- veredicto --------------------------------------------------------------

NIVELES = ["excelente", "muy bueno", "bueno", "justo", "insuficiente"]
GLOBAL = {
    "excelente": "super fluido",
    "muy bueno": "fluido",
    "bueno": "aceptable",
    "justo": "usable con paciencia",
    "insuficiente": "no recomendado",
}


def _clas_lat(ms):
    if ms < 150:
        return "casi instantanea", "excelente"
    if ms < 300:
        return "fluida", "muy bueno"
    if ms < 600:
        return "aceptable", "bueno"
    if ms < 1200:
        return "perceptible", "justo"
    return "lenta", "insuficiente"


def _clas_mem(mb):
    if mb < 250:
        return "liviana", "excelente"
    if mb < 500:
        return "moderada", "muy bueno"
    if mb < 900:
        return "considerable", "bueno"
    if mb < 1500:
        return "pesada", "justo"
    return "muy pesada", "insuficiente"


def _clas_arr(s):
    if s < 3:
        return "rapido", "excelente"
    if s < 8:
        return "agil", "muy bueno"
    if s < 15:
        return "aceptable", "bueno"
    if s < 30:
        return "lento", "justo"
    return "muy lento", "insuficiente"


def _peor(niveles):
    return max(niveles, key=NIVELES.index)


def _ram_rec(pico_mb):
    obj = pico_mb * 2.0
    for limite in (256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384):
        if obj <= limite:
            return limite
    return int(obj)


# -- benchmark --------------------------------------------------------------


def _preparar_temp():
    tmp = tempfile.mkdtemp(prefix="bib_monitor_")
    config.BASE_DIR = tmp  # embeddings.npy va acá
    database.DB_PATH = os.path.join(tmp, "bench.db")
    database.inicializar_db()
    return tmp


def _cargar_corpus(corpus):
    for nombre, paginas in corpus.items():
        database.reemplazar_archivo(nombre, list(enumerate(paginas, start=1)))


def _medir(buscador, consultas, reps):
    """Devuelve lista de dicts con tiempos por fase."""
    filas = []
    for q in consultas:
        for _ in range(reps):
            # BM25 sola
            t0 = time.perf_counter()
            scores_bm25 = buscador.bm25.get_scores(q.lower().split())
            np.argsort(-scores_bm25)  # forzar el costo real del ordenamiento
            t_bm25 = time.perf_counter() - t0

            # Densa sola
            t0 = time.perf_counter()
            from buscador_service import PREFIJO_QUERY

            q_emb = np.array(list(buscador.modelo.embed([PREFIJO_QUERY + q])), dtype=np.float32)[0]
            q_emb = q_emb / (np.linalg.norm(q_emb) or 1.0)
            scores_denso = buscador.embeddings @ q_emb
            np.argsort(-scores_denso)
            t_denso = time.perf_counter() - t0

            # Pipeline completo (RRF incluido)
            t0 = time.perf_counter()
            buscador.buscar(q, top_k=50)
            t_total = time.perf_counter() - t0

            filas.append({"q": q, "bm25": t_bm25, "denso": t_denso, "total": t_total})
    return filas


# -- informe ----------------------------------------------------------------


def _imprimir(r, n_consultas):
    sep = "=" * 60
    print(sep)
    print(" Biblioteca Inteligente - Informe de rendimiento (BM25 + e5-small + RRF)")
    print(sep)
    print()
    print("SISTEMA")
    print("  SO              : %s" % r["sistema"]["so"])
    print("  Arquitectura    : %s" % r["sistema"]["arquitectura"])
    print("  Python          : %s" % r["sistema"]["python"])
    print("  Nucleos         : %d" % r["sistema"]["nucleos"])
    print("  RAM total       : %s" % _fmt_bytes(r["sistema"]["ram_total_bytes"]))
    print()
    print("CORPUS (sintetico, reproducible)")
    print("  Libros          : %d" % r["corpus"]["libros"])
    print("  Paginas         : %d" % r["corpus"]["paginas"])
    print("  Caracteres      : %s" % ("{:,}".format(r["corpus"]["caracteres"]).replace(",", ".")))
    print()
    print("ARRANQUE")
    print("  carga DB        : %.3f s" % r["arranque"]["carga_db_s"])
    print("  BM25            : %.3f s" % r["arranque"]["bm25_s"])
    print("  embeddings      : %.3f s" % r["arranque"]["embeddings_s"])
    print("  carga modelo    : %.3f s" % r["arranque"]["carga_modelo_s"])
    print("  TOTAL           : %.3f s" % r["arranque"]["total_s"])
    print()
    print("BUSQUEDA  (%d consultas x %d reps)" % (n_consultas, r["_reps"]))
    print(
        "  BM25   min/med/max : %.3f / %.3f / %.3f s"
        % (
            r["busqueda"]["bm25_min_ms"] / 1000,
            r["busqueda"]["bm25_med_ms"] / 1000,
            r["busqueda"]["bm25_max_ms"] / 1000,
        )
    )
    print(
        "  densa  min/med/max : %.3f / %.3f / %.3f s"
        % (
            r["busqueda"]["densa_min_ms"] / 1000,
            r["busqueda"]["densa_med_ms"] / 1000,
            r["busqueda"]["densa_max_ms"] / 1000,
        )
    )
    print(
        "  TOTAL  min/med/max : %.3f / %.3f / %.3f s"
        % (
            r["busqueda"]["total_min_ms"] / 1000,
            r["busqueda"]["total_med_ms"] / 1000,
            r["busqueda"]["total_max_ms"] / 1000,
        )
    )
    print()
    print("MEMORIA (RSS)")
    if r["memoria"]["rss_tras_entrenar_mb"] is not None:
        print("  tras entrenar   : %.1f MB" % r["memoria"]["rss_tras_entrenar_mb"])
    if r["memoria"].get("rss_pico_arranque_mb") is not None:
        print("  pico en arranque: %.1f MB" % r["memoria"]["rss_pico_arranque_mb"])
    if r["memoria"]["rss_pico_mb"] is not None:
        print("  pico en busqueda: %.1f MB" % r["memoria"]["rss_pico_mb"])
    print()
    v = r["veredicto"]
    print("VEREDICTO")
    print("  Latencia : %s" % v["latencia"])
    print("  Memoria  : %s" % v["memoria"])
    print("  Arranque : %s" % v["arranque"])
    print("  " + "-" * 60)
    print("  CLASIFICACION GLOBAL : %s" % v["global"])
    print("  RAM recomendada      : %d MB libres" % v["ram_recomendada_mb"])
    print()
    print("-" * 60)
    print("Resumen de una linea (para comparar entre maquinas):")
    print(
        "  [%s %s] %d libros / %d pags | lat %.0f ms | mem %.0f MB | arr %.1f s | %s"
        % (
            r["sistema"]["so"].split()[0],
            r["sistema"]["arquitectura"],
            r["corpus"]["libros"],
            r["corpus"]["paginas"],
            r["busqueda"]["total_med_ms"],
            r["memoria"]["rss_pico_mb"] or 0,
            r["arranque"]["total_s"],
            v["global"],
        )
    )
    print(sep)


def main(argv=None):
    p = argparse.ArgumentParser(prog="monitor.py")
    p.add_argument("--libros", type=int, default=25)
    p.add_argument("--paginas", type=int, default=12)
    p.add_argument("--repeticiones", type=int, default=3)
    p.add_argument("--json", metavar="ARCHIVO")
    args, _ = p.parse_known_args(argv)

    if args.libros < 1 or args.paginas < 1 or args.repeticiones < 1:
        print("ERROR: valores deben ser >= 1", file=sys.stderr)
        return 2

    from buscador_service import PREFIJO_QUERY, BuscadorHibrido  # noqa

    sistema = {
        "so": platform.platform(),
        "arquitectura": platform.machine(),
        "python": platform.python_version(),
        "nucleos": _nucleos(),
        "ram_total_bytes": _ram_total_bytes(),
    }

    tmp = _preparar_temp()
    try:
        corpus = _generar_corpus(args.libros, args.paginas)
        caracteres = sum(len(pg) for pgs in corpus.values() for pg in pgs)

        t0 = time.perf_counter()
        _cargar_corpus(corpus)
        t_carga = time.perf_counter() - t0

        buscador = BuscadorHibrido()
        # Forzar la carga del modelo aunque el hash del corpus coincida
        buscador.modelo = None

        # Medir BM25 y embeddings por separado: corremos entrenar_modelo una vez
        # y después aislamos los tiempos con una corrida manual.
        from rank_bm25 import BM25Okapi

        from buscador_service import MODELO_EMBEDDINGS

        filas = database.obtener_todas_las_paginas()
        paginas = [(f[1], f[2], f[3]) for f in filas]

        t0 = time.perf_counter()
        BM25Okapi([t.lower().split() for _, _, t in paginas])
        t_bm25 = time.perf_counter() - t0

        from fastembed import TextEmbedding

        # Arrancamos el sampler ANTES de cargar el modelo: queremos el pico
        # real de entrenar_modelo, no solo el de la fase de búsqueda.
        muest_arranque = _MuestreadorRSS()
        muest_arranque.start()

        # Una sola instancia, compartida con el buscador. Antes se creaban dos
        # (esta y la interna de entrenar_modelo) y los dos modelos ONNX vivos
        # al mismo tiempo inflaban el RSS.
        t0 = time.perf_counter()
        buscador.modelo = TextEmbedding(model_name=MODELO_EMBEDDINGS, cache_dir=config.MODELOS_DIR)
        t_carga_modelo = time.perf_counter() - t0

        # entrenar_modelo hace el cómputo de embeddings de una sola vez,
        # igual que en producción. Antes se duplicaba acá y el arranque
        # reportado salía al doble del real.
        t0 = time.perf_counter()
        buscador.entrenar_modelo()
        t_emb = time.perf_counter() - t0

        t_total_arr = t_carga + t_bm25 + t_carga_modelo + t_emb

        muest_arranque.detener()
        rss_pico_arranque = muest_arranque.pico

        rss_tras_ent = _rss_bytes()

        muest = _MuestreadorRSS()
        muest.start()
        filas_med = _medir(buscador, CONSULTAS_BENCH, args.repeticiones)
        muest.detener()
        rss_pico = max(muest.pico, _rss_bytes() or 0)

        def _stat(clave):
            vals = [f[clave] * 1000 for f in filas_med]
            return min(vals), statistics.median(vals), max(vals)

        bmin, bmed, bmax = _stat("bm25")
        dmin, dmed, dmax = _stat("denso")
        tmin, tmed, tmax = _stat("total")

        lat_txt, lat_niv = _clas_lat(tmed)
        mem_mb = rss_pico / (1024.0 * 1024.0) if rss_pico else 0.0
        mem_txt, mem_niv = _clas_mem(mem_mb)
        arr_txt, arr_niv = _clas_arr(t_total_arr)
        peor = _peor([lat_niv, mem_niv, arr_niv])

        resultado = {
            "fecha": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "sistema": sistema,
            "corpus": {
                "libros": args.libros,
                "paginas": args.libros * args.paginas,
                "caracteres": caracteres,
            },
            "arranque": {
                "carga_db_s": round(t_carga, 4),
                "bm25_s": round(t_bm25, 4),
                "carga_modelo_s": round(t_carga_modelo, 4),
                "embeddings_s": round(t_emb, 4),
                "total_s": round(t_total_arr, 4),
            },
            "busqueda": {
                "bm25_min_ms": round(bmin, 1),
                "bm25_med_ms": round(bmed, 1),
                "bm25_max_ms": round(bmax, 1),
                "densa_min_ms": round(dmin, 1),
                "densa_med_ms": round(dmed, 1),
                "densa_max_ms": round(dmax, 1),
                "total_min_ms": round(tmin, 1),
                "total_med_ms": round(tmed, 1),
                "total_max_ms": round(tmax, 1),
            },
            "memoria": {
                "rss_tras_entrenar_mb": (
                    round(rss_tras_ent / (1024.0 * 1024.0), 1) if rss_tras_ent else None
                ),
                "rss_pico_arranque_mb": (
                    round(rss_pico_arranque / (1024.0 * 1024.0), 1) if rss_pico_arranque else None
                ),
                "rss_pico_mb": round(mem_mb, 1) if rss_pico else None,
            },
            "veredicto": {
                "latencia": lat_txt,
                "memoria": mem_txt,
                "arranque": arr_txt,
                "global": GLOBAL[peor],
                "ram_recomendada_mb": _ram_rec(mem_mb),
            },
            "_reps": args.repeticiones,
        }

        _imprimir(resultado, len(CONSULTAS_BENCH))
        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(resultado, f, indent=2, ensure_ascii=False)
            print("\nJSON guardado en: %s" % args.json)
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
