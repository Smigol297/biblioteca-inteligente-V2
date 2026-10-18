# -*- coding: utf-8 -*-
"""Tests del BuscadorHibrido (BM25 + e5-small + RRF)."""
import database
from buscador_service import MODELO_ID, BuscadorHibrido

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

        def passage(self, textos):
            return np.array([self._vec(t) for t in textos], dtype=np.float32)

        def query(self, texto):
            return self._vec(texto)

        def _vec(self, t):
            v = np.zeros(self.DIM, dtype=np.float32)
            if "semantico" in t.lower():
                v[0] = 1.0
            else:
                v[1] = 1.0
            return v

    monkeypatch.setattr(buscador_service, "E5", _Semantico)

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


def test_query_y_passage_se_llaman_por_separado(crear_buscador, fake_embedding):
    fake_embedding.textos_recibidos = []
    buscador = crear_buscador(LIBROS)
    # Entrenamiento usa passage()
    assert all(not t.startswith("query:") for t in fake_embedding.textos_recibidos)
    fake_embedding.textos_recibidos = []
    buscador.buscar("fotosintesis")
    # Busqueda usa query() (nanoe5 internamente agrega el prefijo)
    assert fake_embedding.textos_recibidos == ["fotosintesis"]


# ---------- indexado incremental ----------


def test_incremental_no_reembebe_lo_que_no_cambio(crear_buscador, fake_embedding):
    """Segundo entrenar sin cambios no llama a embed."""
    crear_buscador(LIBROS)
    fake_embedding.textos_recibidos = []

    from buscador_service import BuscadorHibrido as B

    b2 = B()
    assert b2.entrenar_modelo() is True
    assert fake_embedding.textos_recibidos == []


def test_incremental_solo_embebe_el_archivo_nuevo(crear_buscador, fake_embedding):
    crear_buscador(LIBROS)
    database.reemplazar_archivo("nuevo.txt", [(1, "contenido flamante")])
    fake_embedding.textos_recibidos = []

    from buscador_service import BuscadorHibrido as B

    b = B()
    b.entrenar_modelo()
    assert fake_embedding.textos_recibidos == ["contenido flamante"]


def test_incremental_reembebe_archivo_modificado(crear_buscador, fake_embedding):
    crear_buscador(LIBROS)
    database.reemplazar_archivo("biologia.txt", [(1, "texto cambiado de biologia")])
    fake_embedding.textos_recibidos = []

    from buscador_service import BuscadorHibrido as B

    b = B()
    b.entrenar_modelo()
    assert fake_embedding.textos_recibidos == ["texto cambiado de biologia"]


def test_incremental_borra_huerfanos_sin_embeber(crear_buscador, fake_embedding):
    crear_buscador(LIBROS)
    database.reemplazar_archivo("cocina.txt", [])  # desaparece la fila
    fake_embedding.textos_recibidos = []

    from buscador_service import BuscadorHibrido as B

    b = B()
    b.entrenar_modelo()
    assert fake_embedding.textos_recibidos == []
    assert "cocina.txt" not in database.obtener_embeddings(MODELO_ID)
    # Y no rompe el ensamblado: quedan las páginas de biologia e historia.
    assert b.embeddings.shape[0] == len(b.paginas)


def test_modelo_id_bumpeado_fuerza_reindexado(crear_buscador, fake_embedding, monkeypatch):
    import buscador_service

    crear_buscador(LIBROS)
    monkeypatch.setattr(buscador_service, "MODELO_ID", "otro-modelo-cualquiera")
    fake_embedding.textos_recibidos = []

    from buscador_service import BuscadorHibrido as B

    b = B()
    b.entrenar_modelo()
    # Se re-embeben todas las páginas con el modelo "nuevo".
    assert len(fake_embedding.textos_recibidos) == sum(len(p) for p in LIBROS.values())


def test_busqueda_anda_despues_de_incremental(crear_buscador):
    b = crear_buscador(LIBROS)
    database.reemplazar_archivo("nuevo.txt", [(1, "revolucion industrial vapor carbon hierro")])
    b.entrenar_modelo()

    assert b.buscar("fotosintesis")[0]["archivo"] == "biologia.txt"
    assert b.buscar("revolucion industrial")[0]["archivo"] == "nuevo.txt"


def test_ensamblado_respeta_orden_de_paginas(crear_buscador):
    b = crear_buscador({"a.txt": ["aaa", "bbb"], "b.txt": ["ccc", "ddd"]})
    assert b.embeddings.shape == (4, 384)
    # Verificar contra b.paginas: para cada i, embeddings[i] tiene que
    # ser el vector de paginas[i]. Recalculamos con el fake y comparamos.
    import numpy as np
    from conftest import _FakeE5

    fake = _FakeE5()
    for i, (_, _, texto) in enumerate(b.paginas):
        esperado = fake.passage([texto])[0]
        np.testing.assert_allclose(b.embeddings[i], esperado, rtol=1e-5)
