from buscador_service import _coleccion_de, _fragmento


def test_coleccion_de_toma_la_carpeta():
    assert _coleccion_de("recetas/torta.pdf") == "recetas"
    assert _coleccion_de("torta.pdf") == ""
    assert _coleccion_de("a/b/c.pdf") == "a"


def test_sin_terminos_devuelve_el_inicio_sin_resaltar():
    texto = "palabra " * 100
    trozos = _fragmento(texto, set())
    assert len(trozos) == 1
    assert trozos[0][1] == 0
    assert trozos[0][0].endswith("…")


def test_texto_corto_sin_coincidencias_no_agrega_puntos_suspensivos():
    assert _fragmento("hola mundo", {"zzz"}) == [["hola mundo", 0]]


def test_resalta_el_termino_y_conserva_las_mayusculas_originales():
    trozos = _fragmento("El Gato duerme sobre el sillon", {"gato"})
    assert ["Gato", 1] in trozos
    assert "".join(t for t, _ in trozos) == "El Gato duerme sobre el sillon"


def test_no_resalta_plurales_ni_variantes():
    # Mismo criterio que la busqueda: "algoritmo" no es "algoritmos"
    trozos = _fragmento("Los algoritmos ordenan datos", {"algoritmo"})
    assert all(resaltado == 0 for _, resaltado in trozos)


def test_centra_la_ventana_en_la_coincidencia():
    texto = ("relleno " * 200) + "aguja " + ("relleno " * 200)
    trozos = _fragmento(texto, {"aguja"}, ancho=300)
    assert trozos[0] == ["… ", 0]
    assert trozos[-1] == [" …", 0]
    assert ["aguja", 1] in trozos
    largo = sum(len(t) for t, _ in trozos)
    assert largo < 330


def test_prefiere_la_ventana_con_mas_terminos_distintos():
    texto = "gato " + ("relleno " * 200) + "gato perro " + ("relleno " * 200)
    trozos = _fragmento(texto, {"gato", "perro"}, ancho=300)
    resaltados = [t for t, r in trozos if r]
    assert sorted(resaltados) == ["gato", "perro"]
