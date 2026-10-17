# -*- coding: utf-8 -*-
"""Compara dos corridas de generar_embeddings.py.

Uso:
    python comparar_embeddings.py --a run_fp32 --b run_o4 --topk 10
"""
import argparse
import json
import sys

import numpy as np


def cargar(prefijo):
    corpus = np.load(prefijo + "_corpus.npy")
    queries = np.load(prefijo + "_queries.npy")
    with open(prefijo + "_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return corpus, queries, meta


def main():
    p = argparse.ArgumentParser(prog="comparar_embeddings.py")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--topk", type=int, default=10)
    args = p.parse_args()

    a_corpus, a_q, a_meta = cargar(args.a)
    b_corpus, b_q, b_meta = cargar(args.b)

    if a_corpus.shape != b_corpus.shape:
        print("ERROR: shapes distintos %s vs %s" % (a_corpus.shape, b_corpus.shape))
        return 1
    if a_meta["consultas"] != b_meta["consultas"]:
        print("ERROR: las consultas no coinciden")
        return 1

    print("Corrida A: %s (%s)" % (a_meta["modelo"], a_meta["archivo_onnx"]))
    print("Corrida B: %s (%s)" % (b_meta["modelo"], b_meta["archivo_onnx"]))
    print("Paginas: %d, Consultas: %d" % (a_corpus.shape[0], len(a_meta["consultas"])))
    print()

    # --- Similitud por pagina ---
    cos = np.sum(a_corpus * b_corpus, axis=1)
    print("=" * 60)
    print("SIMILITUD COSENO POR PAGINA (A vs B)")
    print("=" * 60)
    print("  media  : %.4f" % cos.mean())
    print("  min    : %.4f" % cos.min())
    print("  max    : %.4f" % cos.max())
    print("  <0.90  : %d paginas" % int((cos < 0.90).sum()))
    print("  <0.95  : %d paginas" % int((cos < 0.95).sum()))
    if cos.mean() > 0.9999:
        print("  AVISO: similitud ~1.0. Probablemente ambos .npy son del mismo modelo.")
        print("         Verifica los archivo_onnx en los meta.json de arriba.")
    print()

    # --- Overlap top-k por query ---
    print("=" * 60)
    print("OVERLAP TOP-%d POR CONSULTA" % args.topk)
    print("=" * 60)
    print("%-40s %-10s %s" % ("Consulta", "Overlap", "Detalle"))
    print("-" * 60)
    total_overlap = 0
    for i, q in enumerate(a_meta["consultas"]):
        sa = a_corpus @ a_q[i]
        sb = b_corpus @ b_q[i]
        ta = set(np.argsort(-sa)[: args.topk].tolist())
        tb = set(np.argsort(-sb)[: args.topk].tolist())
        overlap = len(ta & tb)
        total_overlap += overlap
        detalle = ""
        if overlap < args.topk:
            detalle = "A-solo: %d, B-solo: %d" % (len(ta - tb), len(tb - ta))
        print("%-40s %d/%d      %s" % (q[:39], overlap, args.topk, detalle))
    print("-" * 60)
    promedio = total_overlap / (len(a_meta["consultas"]) * args.topk)
    print("Promedio: %.2f%%" % (100.0 * promedio))
    return 0


if __name__ == "__main__":
    sys.exit(main())
