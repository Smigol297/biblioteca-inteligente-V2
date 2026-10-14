import buscador_service
from buscador_service import BuscadorLocal

# Se guarda el metodo real antes de que el fixture autouse lo reemplace
_TRADUCIR_REAL = BuscadorLocal._traducir_consulta

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


def test_entrenar_sin_datos_devuelve_false(entorno):
    assert BuscadorLocal().entrenar_modelo() is False


def test_buscar_sin_entrenar_devuelve_lista_vacia(entorno):
    assert BuscadorLocal().buscar("hola") == []


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


class _TraductorFalso:
    def __init__(self, from_lang, to_lang):
        pass

    def translate(self, texto):
        return "cat"


def test_traducir_concatena_ambos_idiomas(monkeypatch):
    monkeypatch.setattr(BuscadorLocal, "_traducir_consulta", _TRADUCIR_REAL)
    monkeypatch.setattr(buscador_service, "Translator", _TraductorFalso)
    assert BuscadorLocal()._traducir_consulta("gato") == "gato cat"


def test_traducir_igual_a_la_consulta_no_duplica(monkeypatch):
    class _Igual(_TraductorFalso):
        def translate(self, texto):
            return texto.upper()

    monkeypatch.setattr(BuscadorLocal, "_traducir_consulta", _TRADUCIR_REAL)
    monkeypatch.setattr(buscador_service, "Translator", _Igual)
    assert BuscadorLocal()._traducir_consulta("hotel") == "hotel"


def test_traducir_sin_internet_devuelve_la_consulta(monkeypatch):
    class _SinRed(_TraductorFalso):
        def translate(self, texto):
            raise ConnectionError("sin red")

    monkeypatch.setattr(BuscadorLocal, "_traducir_consulta", _TRADUCIR_REAL)
    monkeypatch.setattr(buscador_service, "Translator", _SinRed)
    assert BuscadorLocal()._traducir_consulta("gato") == "gato"
