# -*- coding: utf-8 -*-
"""Compara FP32 vs INT8 O4 sobre el mismo corpus.

Mide:
  1. Similitud coseno por pagina entre los embeddings de cada modelo.
  2. Overlap del top-10 para varias consultas.

Uso:
    python comparar_modelos.py
    python comparar_modelos.py --limite 500
    python comparar_modelos.py --consultas "fotosintesis,torta de chocolate"

Interpretacion:
  - Similitud media > 0.95  -> INT8 casi indistinguible de FP32.
  - Similitud media 0.90-0.95 -> perdida chica pero real.
  - Similitud media < 0.90  -> INT8 cambia el espacio; revisar resultados.
  - Overlap top-10 = 10/10  -> ranking identico para esa consulta.
  - Overlap < 8/10          -> INT8 reordena cosas importantes.
"""
import argparse
import sys

import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

import config
import database

MODELO_HF = "intfloat/multilingual-e5-small"
PREFIJO_PASSAGE = "passage: "
PREFIJO_QUERY = "query: "
BATCH_SIZE = 64
DIM = 384

# Edita esta lista con consultas REALES de tu corpus.
CONSULTAS_DEFAULT = [
    "algoritmo de ordenamiento",
    "memoria del sistema",
    "consulta a la base de datos",
    "red y servidor",
    "procesamiento de archivos",
]


def _registrar(nombre_id, model_file):
    """Registra un modelo custom bajo un nombre distinto para no pisar el de buscador_service."""
    try:
        TextEmbedding.add_custom_model(
            model=nombre_id,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=MODELO_HF),
            dim=DIM,
            model_file=model_file,
        )
    except Exception as e:
        # Si ya estaba registrado, fastembed tira; lo ignoramos.
        print("  (aviso al registrar %s: %s)" % (nombre_id, e))


def _embeber(modelo, textos):
    """Devuelve (n, DIM) float32 normalizado L2."""
    arr = np.empty((len(textos), DIM), dtype=np.float32)
    for i, v in enumerate(modelo.embed(textos, batch_size=BATCH_SIZE)):
        arr[i] = v
    normas = np.linalg.norm(arr, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    arr /= normas
    return arr


def _embeber_query(modelo, q):
    v = np.asarray(next(iter(modelo.embed([PREFIJO_QUERY + q], batch_size=1))), dtype=np.float32)
    n = float(np.linalg.norm(v)) or 1.0
    return v / n


def main():
    p = argparse.ArgumentParser(prog="comparar_modelos.py")
    p.add_argument("--limite", type=int, default=0, help="Maximo de paginas a comparar (0=todas)")
    p.add_argument("--consultas", type=str, default="", help="Consultas separadas por coma")
    p.add_argument("--topk", type=int, default=10)
    args = p.parse_args()

    filas = database.obtener_todas_las_paginas()
    if not filas:
        print("ERROR: no hay paginas en la DB. Indexa algo primero.")
        return 1
    if args.limite:
        filas = filas[: args.limite]

    textos = [PREFIJO_PASSAGE + f[3] for f in filas]
    print("Corpus: %d paginas" % len(textos))

    print("\nRegistrando modelos (nombres distintos para no pisar buscador_service)...")
    _registrar("comparar/e5-fp32", "onnx/model.onnx")
    _registrar("comparar/e5-o4", "onnx/model_O4.onnx")

    print("\nCargando FP32...")
    m_fp32 = TextEmbedding(model_name="comparar/e5-fp32", cache_dir=config.MODELOS_DIR)
    print("Embeiendo FP32...")
    emb_fp32 = _embeber(m_fp32, textos)

    print("\nCargando INT8 O4...")
    m_o4 = TextEmbedding(model_name="comparar/e5-o4", cache_dir=config.MODELOS_DIR)
    print("Embeiendo INT8 O4...")
    emb_o4 = _embeber(m_o4, textos)

    # --- Comparacion 1: similitud por pagina ---
    cos = np.sum(emb_fp32 * emb_o4, axis=1)
    print("\n" + "=" * 60)
    print("SIMILITUD COSENO FP32 vs INT8 (por pagina)")
    print("=" * 60)
    print("  media  : %.4f" % cos.mean())
    print("  min    : %.4f" % cos.min())
    print("  max    : %.4f" % cos.max())
    print("  <0.90  : %d paginas" % int((cos < 0.90).sum()))
    print("  <0.95  : %d paginas" % int((cos < 0.95).sum()))

    # --- Comparacion 2: rankings por consulta ---
    consultas = (
        [c.strip() for c in args.consultas.split(",") if c.strip()]
        if args.consultas
        else CONSULTAS_DEFAULT
    )
    print("\n" + "=" * 60)
    print("OVERLAP TOP-%d POR CONSULTA" % args.topk)
    print("=" * 60)
    print("%-45s %s" % ("Consulta", "Overlap"))
    print("-" * 60)
    for q in consultas:
        s32 = emb_fp32 @ _embeber_query(m_fp32, q)
        s_o4 = emb_o4 @ _embeber_query(m_o4, q)
        t32 = set(np.argsort(-s32)[: args.topk].tolist())
        t_o4 = set(np.argsort(-s_o4)[: args.topk].tolist())
        overlap = len(t32 & t_o4)
        extra = ""
        if overlap < args.topk:
            solo32 = len(t32 - t_o4)
            solo_o4 = len(t_o4 - t32)
            extra = " (FP32-solo: %d, INT8-solo: %d)" % (solo32, solo_o4)
        print("%-45s %d/%d%s" % (q[:44], overlap, args.topk, extra))

    print("\nListo. Si el overlap es consistentemente <%d, INT8 no es opcion." % (args.topk - 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
