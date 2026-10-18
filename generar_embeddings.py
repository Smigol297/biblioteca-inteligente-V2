# -*- coding: utf-8 -*-
"""Genera embeddings del corpus y de las queries con UN solo modelo.

Se corre dos veces: una para FP32 y una para INT8. Cada corrida en su
propio proceso, para que no haya ambiguedad sobre que archivo ONNX carga.

Uso:
    python generar_embeddings.py --modelo fp32 --salida run_fp32 --consultas "a,b,c"
    python generar_embeddings.py --modelo o4   --salida run_o4   --consultas "a,b,c"
"""
import argparse
import json
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
ARCHIVOS = {
    "fp32": "onnx/model.onnx",
    "o4": "onnx/model_O4.onnx",
}


def main():
    p = argparse.ArgumentParser(prog="generar_embeddings.py")
    p.add_argument("--modelo", required=True, choices=list(ARCHIVOS))
    p.add_argument("--salida", required=True, help="Prefijo de los archivos de salida")
    p.add_argument("--limite", type=int, default=0)
    p.add_argument("--consultas", required=True, help="Consultas separadas por coma")
    args = p.parse_args()

    filas = database.obtener_todas_las_paginas()
    if not filas:
        print("ERROR: no hay paginas en la DB.")
        return 1
    if args.limite:
        filas = filas[: args.limite]
    textos = [PREFIJO_PASSAGE + f[3] for f in filas]
    consultas = [c.strip() for c in args.consultas.split(",") if c.strip()]
    print("[%s] Corpus: %d paginas, %d consultas" % (args.modelo, len(textos), len(consultas)))

    archivo_onnx = ARCHIVOS[args.modelo]
    print("[%s] Registrando modelo custom apuntando a: %s" % (args.modelo, archivo_onnx))
    TextEmbedding.add_custom_model(
        model="bib-e5",
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf=MODELO_HF),
        dim=DIM,
        model_file=archivo_onnx,
    )

    print("[%s] Cargando modelo..." % args.modelo)
    modelo = TextEmbedding(model_name="bib-e5", cache_dir=config.MODELOS_DIR)

    print("[%s] Embeiendo corpus..." % args.modelo)
    corpus = np.empty((len(textos), DIM), dtype=np.float32)
    for i, v in enumerate(modelo.embed(textos, batch_size=BATCH_SIZE)):
        corpus[i] = v
    normas = np.linalg.norm(corpus, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    corpus /= normas

    print("[%s] Embeiendo consultas..." % args.modelo)
    queries = np.empty((len(consultas), DIM), dtype=np.float32)
    for i, q in enumerate(consultas):
        v = np.asarray(
            next(iter(modelo.embed([PREFIJO_QUERY + q], batch_size=1))), dtype=np.float32
        )
        n = float(np.linalg.norm(v)) or 1.0
        queries[i] = v / n

    np.save(args.salida + "_corpus.npy", corpus)
    np.save(args.salida + "_queries.npy", queries)
    with open(args.salida + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "modelo": args.modelo,
                "archivo_onnx": archivo_onnx,
                "consultas": consultas,
                "paginas": len(textos),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(
        "[%s] Guardado en %s_{corpus,queries}.npy y %s_meta.json"
        % (args.modelo, args.salida, args.salida)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
