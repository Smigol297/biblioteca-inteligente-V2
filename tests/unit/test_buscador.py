# -*- coding: utf-8 -*-
"""Tests del BuscadorHibrido (BM25 + e5-small + RRF)."""
from buscador_service import BuscadorHibrido

LIBROS = {
    "biologia.txt": [
        "La fotosintesis convierte la luz solar en energia quimica dentro de las plantas",
        "Las mitocondrias producen energia para la celula mediante la respiracion celular",
    ],
    "cocina.txt": [
        "Para la torta se mezclan harina huevos azucar y manteca en un bowl grande",
        "El asado se cocina lento sobre las brasas con sal gruesa",
    ],
    "historia.txt": [
        "La revolucion de mayo cambio el gobierno del virreinato del Rio de la Plata",
    ],
}


# ---------- entrenamiento ----------


def test_entrenar_sin_datos_devuelve_false(entorno):
    assert BuscadorHibrido().entrenar_modelo() is False


def test_buscar_sin_entrenar_devuelve_lista_vacia(entorno):
    assert BuscadorHibrido().buscar("hola") == []


# ---------- busqueda basica ----------


def test_encuentra_la_pagina_correcta(crear_buscador):
    buscador = crear_buscador(LIBROS)
    resultados = buscador.buscar("fotosintesis")
    assert resultados[0]["archivo"] == "biologia.txt"
    assert resultados[0]["pagina"] == 1
    assert {"archivo", "pagina", "texto", "fragmento", "score"} <= set(resultados[0])


def test_resalta_el_termino_buscado(crear_buscador):
    resultados = crear_buscador(LIBROS).buscar("fotosintesis")
    assert ["fotosintesis", 1] in [[t.lower(), r] for t, r in resultados[0]["fragmento"]]


def test_texto_es_el_fragmento_plano(crear_buscador):
    resultado = crear_buscador(LIBROS).buscar("torta")[0]
    assert resultado["texto"] == "".join(t for t, _ in resultado["fragmento"])


def test_consulta_sin_coincidencias_devuelve_vacio(crear_buscador):
    assert crear_buscador(LIBROS).buscar("zzzzqwerty") == []


def test_top_k_limita_la_cantidad(crear_buscador):
    buscador = crear_buscador(LIBROS)
    assert len(buscador.buscar("energia", top_k=1)) == 1


def test_agrupar_deja_una_pagina_por_libro(crear_buscador):
    buscador = crear_buscador(LIBROS)
    sin_agrupar = [r["archivo"] for r in buscador.buscar("energia", umbral_rel=0.0)]
    agrupado = [r["archivo"] for r in buscador.buscar("energia", umbral_rel=0.0, agrupar=True)]
    assert sin_agrupar.count("biologia.txt") == 2
    assert agrupado.count("biologia.txt") == 1


# ---------- colecciones ----------


def test_filtro_por_coleccion(crear_buscador):
    texto = ["El elefante africano es el animal terrestre mas grande del planeta"]
    buscador = crear_buscador(
        {
            "zoologia/animales.txt": texto,
            "infantil/cuentos.txt": texto,
            "suelto.txt": texto,
            "otro.txt": ["Contenido sobre un tema completamente distinto de aquel"],
        }
    )
    todas = {r["archivo"] for r in buscador.buscar("elefante")}
    assert todas == {"zoologia/animales.txt", "infantil/cuentos.txt", "suelto.txt"}
    solo_zoo = [r["archivo"] for r in buscador.buscar("elefante", coleccion="zoologia")]
    assert solo_zoo == ["zoologia/animales.txt"]
    sin_coleccion = {r["archivo"] for r in buscador.buscar("elefante", coleccion="")}
    assert sin_coleccion == {"suelto.txt"}
    assert buscador.buscar("elefante", coleccion="inexistente") == []


# ---------- terminos resaltables ----------


def test_terminos_resaltables_descarta_palabras_comunes(crear_buscador):
    buscador = crear_buscador(
        {
            "a.txt": ["de la quimica organica"],
            "b.txt": ["de la fisica moderna"],
            "c.txt": ["de la historia antigua"],
            "d.txt": ["de la poesia clasica"],
        }
    )
    assert buscador._terminos_resaltables("de quimica") == {"quimica"}
    # Si solo hay palabras comunes, se usan igual (mejor eso que no resaltar nada)
    assert buscador._terminos_resaltables("de la") == {"de", "la"}


# ---------- RRF: fusion de los dos rankings ----------


def test_rrf_fusiona_bm25_y_denso(crear_buscador, monkeypatch):
    """Con un fake que favorece textos con 'semantico' y BM25 favoreciendo
    'algoritmo', los dos documentos tienen que aparecer en el resultado.

    Sin RRF, uno de los dos quedaria afuera segun quien ganara la pelea.
    """
    import numpy as np

    import buscador_service

    class _Semantico:
        DIM = 384

        def __init__(self, *a, **kw):
            pass

        def embed(self, textos, batch_size=None, **kwargs):
            # batch_size y kwargs se ignoran: el fake produce lo mismo
            # sin importar cómo se trocee la entrada.
            for t in textos:
                v = np.zeros(self.DIM, dtype=np.float32)
                if "semantico" in t.lower():
                    v[0] = 1.0
                else:
                    v[1] = 1.0
                yield v

    monkeypatch.setattr(buscador_service, "TextEmbedding", _Semantico)

    buscador = crear_buscador(
        {
            "lexico.txt": ["algoritmo algoritmo algoritmo algoritmo aparece aqui"],
            "semantico.txt": ["una palabra semantico nada mas"],
            "ruido.txt": ["cocina recetas torta de chocolate"],
        }
    )
    resultados = buscador.buscar("algoritmo semantico")
    nombres = [r["archivo"] for r in resultados]

    # lexico gana en BM25 (4 apariciones de 'algoritmo'),
    # semantico gana en densa (el fake pone coseno 1 con la consulta).
    # Ambos deben sobrevivir al umbral y aparecer.
    assert "lexico.txt" in nombres
    assert "semantico.txt" in nombres
    # ruido no comparte nada con la consulta en ningun ranking.
    assert "ruido.txt" not in nombres


# ---------- prefijos de e5 ----------


def test_prefijos_query_y_passage(crear_buscador, fake_embedding):
    """e5 requiere 'passage: ' en los documentos y 'query: ' en la consulta.
    Omitirlos degrada el recall.
    """
    fake_embedding.textos_recibidos = []
    buscador = crear_buscador(LIBROS)

    # Durante el entrenamiento, todos los textos van con 'passage: '.
    assert fake_embedding.textos_recibidos
    assert all(t.startswith("passage: ") for t in fake_embedding.textos_recibidos)

    # Durante la busqueda, la consulta va con 'query: '.
    fake_embedding.textos_recibidos = []
    buscador.buscar("fotosintesis")
    assert fake_embedding.textos_recibidos == ["query: fotosintesis"]
