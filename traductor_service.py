"""Traducción local ES→EN con CTranslate2 INT8. Sin PyTorch, sin red."""

import os

import ctranslate2
import sentencepiece as spm
from huggingface_hub import snapshot_download

_MODELO_ID = "Prukario/opus-mt-es-en-ct2-int8"
_translator = None
_sp_src = None
_sp_tgt = None


def _cargar():
    """Carga lazy: solo la primera vez que se necesita traducir."""
    global _translator, _sp_src, _sp_tgt
    if _translator is not None:
        return
    path = snapshot_download(_MODELO_ID)
    _translator = ctranslate2.Translator(
        path,
        device="cpu",
        compute_type="int8",
        inter_threads=1,  # limitar paralelismo entre peticiones
        intra_threads=2,  # limitar hilos por petición
    )
    _sp_src = spm.SentencePieceProcessor(model_file=os.path.join(path, "source.spm"))
    _sp_tgt = spm.SentencePieceProcessor(model_file=os.path.join(path, "target.spm"))


def _parece_espanol(texto):
    """Heurística: tildes/ñ o palabras comunes en español."""
    if any(c in texto.lower() for c in "áéíóúñü¿¡"):
        return True
    palabras_es = {"cómo", "qué", "para", "con", "los", "las", "una", "del", "que", "por"}
    return len(set(texto.lower().split()) & palabras_es) >= 2


def traducir_si_espanol(texto):
    """Devuelve el texto en inglés si parece español; si no, el original."""
    if not _parece_espanol(texto):
        return texto
    try:
        _cargar()
        tokens = _sp_src.encode(texto, out_type=str) + ["</s>"]
        salida = _translator.translate_batch([tokens], beam_size=2)[0].hypotheses[0]
        if "</s>" in salida:
            salida = salida[: salida.index("</s>")]
        return _sp_tgt.decode(salida)
    except Exception:
        return texto  # fallback: query original
