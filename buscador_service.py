# -*- coding: utf-8 -*-
"""Buscador hibrido: BM25 (lexico) + multilingual-e5-small (semantico) con RRF.

Reemplaza al BuscadorLocal basado en TF-IDF. Sin traduccion (e5 es multilingue)
y sin dependencia de internet en tiempo de busqueda.
"""
import hashlib
import os
import re

import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
from rank_bm25 import BM25Okapi

# --- Registro manual de e5-small (no está en el catálogo de esta versión) ---
# fastembed 0.9.0 solo incluye multilingual-e5-large. Registramos el small
# explícitamente para que TextEmbedding lo reconozca.
# Ver: https://github.com/qdrant/fastembed/issues/123
try:
    TextEmbedding.add_custom_model(
        model="intfloat/multilingual-e5-small",
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf="intfloat/multilingual-e5-small"),
        dim=384,
        # INT8 cuantizado portable: ~113 MB en disco (vs ~470 MB del FP32).
        # Se eligio model_O4 y no model_qint8_avx512_vnni porque este equipo
        # (y probablemente el del usuario final) no tiene AVX-512 + VNNI.
        # Cargar la variante avx512 sin esos flags puede ser MAS LENTO que
        # el FP32. Verificado en Debian x86_64 sin AVX-512.
        model_file="onnx/model_O4.onnx",
    )
    print("Modelo e5-small registrado manualmente.")
except Exception as e:
    print("Aviso: no se pudo registrar e5-small: %s" % e)
# ---------------------------------------------------------------------------

import config
import database

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MODELO_EMBEDDINGS = "intfloat/multilingual-e5-small"
RRF_K = 60
UMBRAL_REL = 0.3
UMBRAL_DENSO_MIN = 0.4  # coseno minimo para considerar que hay match semantico
ANCHO_FRAGMENTO = 300
TOKEN_RE = re.compile(r"\w\w+")
PREFIJO_PASSAGE = "passage: "
PREFIJO_QUERY = "query: "
EMBED_DIM = 384
# Curva medida en Debian x86_64, 4 nucleos, corpus 300 pags:
#   32  -> 112 s / 1.46 GB
#   64  ->  85 s / 1.50 GB   <- punto elegido
#   128 ->  81 s / 2.14 GB   (codo: +643 MB por -4 s)
#   256 ->  54 s / 5.80 GB
BATCH_SIZE = 64
# ---------------------------------------------------------------------------
# Helpers (fragmento, coleccion)
# ---------------------------------------------------------------------------


def _coleccion_de(archivo):
    if "/" in archivo:
        return archivo.split("/", 1)[0]
    return ""


def _fragmento(texto, terminos, ancho=ANCHO_FRAGMENTO):
    """Ventana de ~ancho caracteres centrada en los terminos. Lista [trozo, resaltado]."""
    if not terminos:
        trozo = texto[:ancho]
        if len(texto) > ancho:
            trozo = trozo.rsplit(" ", 1)[0] + " …"
        return [[trozo, 0]]

    apariciones = []
    for t in terminos:
        for m in re.finditer(r"\b" + re.escape(t) + r"\b", texto, re.IGNORECASE):
            apariciones.append((m.start(), t))
            if len(apariciones) >= 50:
                break
        if len(apariciones) >= 50:
            break

    if not apariciones:
        trozo = texto[:ancho]
        if len(texto) > ancho:
            trozo = trozo.rsplit(" ", 1)[0] + " …"
        return [[trozo, 0]]

    mejor_ini, mejor_fin, mejor_score = 0, min(len(texto), ancho), (-1, -1)
    for pos, _ in apariciones:
        ini = max(0, pos - ancho // 2)
        fin = min(len(texto), ini + ancho)
        if fin - ini < ancho:
            ini = max(0, fin - ancho)
        distintos, total = set(), 0
        for p, t in apariciones:
            if ini <= p < fin:
                distintos.add(t)
                total += 1
        score = (len(distintos), total)
        if score > mejor_score:
            mejor_score = score
            mejor_ini, mejor_fin = ini, fin

    ini, fin = mejor_ini, mejor_fin
    if ini > 0:
        esp = texto.rfind(" ", 0, ini + 1)
        if esp != -1:
            ini = esp + 1
    if fin < len(texto):
        esp = texto.find(" ", fin)
        if esp != -1:
            fin = esp

    prefijo = "… " if ini > 0 else ""
    sufijo = " …" if fin < len(texto) else ""
    recorte = texto[ini:fin]

    trozos = []
    if prefijo:
        trozos.append([prefijo, 0])
    pos = 0
    patron = r"\b(" + "|".join(re.escape(t) for t in terminos) + r")\b"
    for m in re.finditer(patron, recorte, re.IGNORECASE):
        if m.start() > pos:
            trozos.append([recorte[pos : m.start()], 0])
        trozos.append([m.group(0), 1])
        pos = m.end()
    if pos < len(recorte):
        trozos.append([recorte[pos:], 0])
    if sufijo:
        trozos.append([sufijo, 0])
    return trozos


# ---------------------------------------------------------------------------
# Buscador hibrido
# ---------------------------------------------------------------------------


class BuscadorHibrido:
    """BM25 + e5-small, fusionados con RRF. Interfaz compatible con BuscadorLocal."""

    def __init__(self):
        self.paginas = []
        self.colecciones = np.array([])
        self.bm25 = None
        self.embeddings = None
        self.modelo = None
        self._entrenado = False
        self.corpus_tokens = []

    # -- entrenamiento ------------------------------------------------------

    def entrenar_modelo(self):
        filas = database.obtener_todas_las_paginas()
        if not filas:
            self._entrenado = False
            return False

        self.paginas = [(f[1], f[2], f[3]) for f in filas]
        self.colecciones = np.array([_coleccion_de(a) for a, _, _ in self.paginas])

        # BM25: tokenizacion simple. Guardamos el corpus tokenizado nosotros
        # porque BM25Okapi no lo expone como atributo publico.
        self.corpus_tokens = [t.lower().split() for _, _, t in self.paginas]
        self.bm25 = BM25Okapi(self.corpus_tokens)

        # El modelo se carga siempre: buscar() lo necesita para embeber la
        # consulta, aunque los embeddings del corpus se reusen del .npy.
        if self.modelo is None:
            self.modelo = TextEmbedding(
                model_name=MODELO_EMBEDDINGS,
                cache_dir=config.MODELOS_DIR,
            )

        # Embeddings: reusar si el corpus no cambio
        ruta_emb = os.path.join(config.BASE_DIR, "embeddings.npy")
        hash_actual = self._hash_corpus()
        if os.path.exists(ruta_emb):
            try:
                datos = np.load(ruta_emb, allow_pickle=True).item()
                if datos.get("hash") == hash_actual and datos["emb"].shape[0] == len(self.paginas):
                    self.embeddings = datos["emb"]
                    self._entrenado = True
                    return True
            except Exception:
                pass

        # e5 requiere "passage: " para los documentos
        textos = [PREFIJO_PASSAGE + t for _, _, t in self.paginas]
        n = len(textos)
        # Prealocamos float32 y copiamos cada vector: evita materializar la
        # lista intermedia de ndarrays y mantiene el dtype de punta a punta.
        self.embeddings = np.empty((n, EMBED_DIM), dtype=np.float32)
        for i, vec in enumerate(self.modelo.embed(textos, batch_size=BATCH_SIZE)):
            self.embeddings[i] = vec
            
        assert i == n - 1, "fastembed devolvio %d vectores, esperaba %d" % (i + 1, n)
        # Normalizar in-place: mismo resultado, sin array temporal nuevo.
        normas = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normas[normas == 0] = 1.0
        self.embeddings /= normas
        np.save(ruta_emb, {"hash": hash_actual, "emb": self.embeddings})
        self._entrenado = True
        return True

    def _hash_corpus(self):
        h = hashlib.md5()
        for a, p, t in self.paginas:
            h.update(("%s|%s|%s" % (a, p, t)).encode("utf-8", errors="replace"))
        return h.hexdigest()

    # -- busqueda -----------------------------------------------------------

    def buscar(self, query, top_k=100, agrupar=False, coleccion=None, umbral_rel=None):
        if not self._entrenado or not query.strip():
            return []

        # BM25
        tokens = query.lower().split()
        scores_bm25 = np.array(self.bm25.get_scores(tokens))
        # Mascara de overlap: docs que comparten al menos un token con la consulta.
        # No alcanza con "score > 0" porque con terminos frecuentes el IDF de
        # BM25 es negativo y un doc relevante termina con score negativo.
        tokens_set = set(tokens)
        mask_bm25 = np.array([bool(tokens_set & set(doc)) for doc in self.corpus_tokens])

        # e5 requiere "query: " para la consulta
        if self.modelo is None:
            self.modelo = TextEmbedding(
                model_name=MODELO_EMBEDDINGS,
                cache_dir=config.MODELOS_DIR,
            )
        q_emb = np.asarray(
            next(iter(self.modelo.embed([PREFIJO_QUERY + query], batch_size=1))),
            dtype=np.float32,
        )
        q_emb = q_emb / (np.linalg.norm(q_emb) or 1.0)
        scores_denso = self.embeddings @ q_emb

        # Filtro por coleccion
        if coleccion is not None:
            mascara = self.colecciones == ("" if coleccion == "" else coleccion)
            scores_bm25 = np.where(mascara, scores_bm25, -np.inf)
            scores_denso = np.where(mascara, scores_denso, -np.inf)
            mask_bm25 = mask_bm25 & mascara

        # Rankings
        idx_bm25 = np.argsort(-scores_bm25)
        idx_denso = np.argsort(-scores_denso)

        # RRF: entra al ranking quien tenga match lexico (mask_bm25) o semantico
        # (coseno >= UMBRAL_DENSO_MIN).
        rrf = {}
        for rank, idx in enumerate(idx_bm25, start=1):
            if mask_bm25[idx]:
                rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (RRF_K + rank)
        for rank, idx in enumerate(idx_denso, start=1):
            if scores_denso[idx] >= UMBRAL_DENSO_MIN:
                rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (RRF_K + rank)

        if not rrf:
            return []

        mejor = max(rrf.values())
        _umbral_rel = UMBRAL_REL if umbral_rel is None else umbral_rel
        umbral = mejor * _umbral_rel
        candidatos = sorted(rrf.items(), key=lambda kv: -kv[1])
        terminos = self._terminos_resaltables(query)

        resultados = []
        vistos = set()
        for idx, score in candidatos:
            if score < umbral:
                continue
            archivo, pagina, texto = self.paginas[idx]
            if agrupar:
                if archivo in vistos:
                    continue
                vistos.add(archivo)
            frag = _fragmento(texto, terminos)
            resultados.append(
                {
                    "archivo": archivo,
                    "pagina": pagina,
                    "texto": "".join(t for t, _ in frag),
                    "fragmento": frag,
                    "score": float(score),
                }
            )
            if len(resultados) >= top_k:
                break
        return resultados

    def _terminos_resaltables(self, query):
        """Tokens de la consulta que existen en el corpus y no son ubicuos.

        Antes se filtraba por idf > 1.7; ahora se filtra por frecuencia de
        documento: un termino que aparece en mas del 60% de las paginas es
        demasiado comun para resaltarlo. Si todos son comunes, se usan
        igual (mejor eso que no resaltar nada).
        """
        if self.bm25 is None:
            return set()
        tokens = TOKEN_RE.findall(query.lower())
        n_docs = len(self.corpus_tokens)
        if n_docs == 0:
            return set(tokens)
        vocab_filtrado = set()
        for t in tokens:
            apariciones = sum(1 for doc in self.corpus_tokens if t in doc)
            if apariciones > 0 and apariciones / float(n_docs) < 0.6:
                vocab_filtrado.add(t)
        return vocab_filtrado or set(tokens)
