import io
import types

import pytest

import database

BIOLOGIA = "La fotosintesis convierte la luz solar en energia quimica dentro de las plantas"
COCINA = "Para la torta se mezclan harina huevos azucar y manteca en un bowl grande"
HISTORIA = "La revolucion de mayo cambio el gobierno del virreinato del Rio de la Plata"


def subir(cliente, nombre, contenido, **extra):
    if isinstance(contenido, str):
        contenido = contenido.encode("utf-8")
    datos = {"file": (io.BytesIO(contenido), nombre)}
    datos.update(extra)
    return cliente.post("/subir", data=datos, content_type="multipart/form-data")


@pytest.fixture
def biblioteca(cliente):
    """Cliente con tres libros ya subidos."""
    for nombre, texto in [
        ("biologia.txt", BIOLOGIA),
        ("cocina.txt", COCINA),
        ("historia.txt", HISTORIA),
    ]:
        assert subir(cliente, nombre, texto).get_json()["exito"] is True
    return cliente


# ---------- subir / buscar ----------


def test_subir_y_buscar(biblioteca):
    resultados = biblioteca.get("/buscar?q=fotosintesis").get_json()
    assert resultados[0]["archivo"] == "biologia.txt"
    assert resultados[0]["pagina"] == 1
    assert any(resaltado for _, resaltado in resultados[0]["fragmento"])


def test_buscar_sin_consulta_devuelve_lista_vacia(biblioteca):
    assert biblioteca.get("/buscar?q=").get_json() == []
    assert biblioteca.get("/buscar").get_json() == []


def test_buscar_parametro_n(biblioteca):
    assert len(biblioteca.get("/buscar?q=la&n=1").get_json()) == 1
    assert len(biblioteca.get("/buscar?q=la&n=0").get_json()) == 1  # se limita a minimo 1
    assert biblioteca.get("/buscar?q=la&n=abc").status_code == 200  # valor invalido: usa 100


def test_subir_el_mismo_libro_dos_veces_no_duplica(cliente):
    subir(cliente, "libro.txt", BIOLOGIA)
    subir(cliente, "libro.txt", BIOLOGIA)
    assert len(database.obtener_todas_las_paginas()) == 1


def test_subir_sin_archivo(cliente):
    respuesta = cliente.post("/subir", data={}, content_type="multipart/form-data")
    assert respuesta.get_json()["exito"] is False


def test_subir_formato_no_permitido(cliente):
    respuesta = subir(cliente, "virus.exe", "contenido")
    assert respuesta.get_json()["exito"] is False
    assert "Formato no permitido" in respuesta.get_json()["mensaje"]


def test_subir_archivo_sin_texto_informa_el_error(cliente):
    respuesta = subir(cliente, "vacio.txt", "corto")
    assert respuesta.get_json()["exito"] is False


def test_subir_pdf_y_buscar(cliente, crear_pdf, tmp_path):
    ruta = crear_pdf(tmp_path / "manual.pdf", ["Capitulo sobre termodinamica y calor en los gases"])
    with open(str(ruta), "rb") as f:
        respuesta = subir(cliente, "manual.pdf", f.read())
    assert respuesta.get_json()["exito"] is True
    resultados = cliente.get("/buscar?q=termodinamica").get_json()
    assert resultados[0]["archivo"] == "manual.pdf"
    ver = cliente.get("/ver/manual.pdf")
    assert ver.status_code == 200
    assert ver.mimetype == "application/pdf"


# ---------- entrenar=0 / reentrenar ----------


def test_entrenar_cero_no_reentrena_hasta_pedirlo(cliente):
    subir(cliente, "biologia.txt", BIOLOGIA, entrenar="0")
    assert cliente.get("/buscar?q=fotosintesis").get_json() == []
    assert cliente.post("/reentrenar").get_json() == {"exito": True}
    assert cliente.get("/buscar?q=fotosintesis").get_json()[0]["archivo"] == "biologia.txt"


# ---------- colecciones ----------


def test_coleccion_guarda_en_subcarpeta_y_se_filtra(cliente, entorno):
    subir(cliente, "torta.txt", COCINA, coleccion="recetas")
    subir(cliente, "suelto.txt", COCINA)
    assert (entorno / "recetas" / "torta.txt").exists()

    info = cliente.get("/colecciones").get_json()
    assert info == {"colecciones": ["recetas"], "hay_sin": True}

    en_recetas = cliente.get("/buscar?q=torta&coleccion=recetas").get_json()
    assert [r["archivo"] for r in en_recetas] == ["recetas/torta.txt"]
    sin_coleccion = cliente.get("/buscar?q=torta&coleccion=").get_json()
    assert [r["archivo"] for r in sin_coleccion] == ["suelto.txt"]
    assert cliente.get("/buscar?q=torta&coleccion=otra").get_json() == []
    assert len(cliente.get("/buscar?q=torta").get_json()) == 2


def test_colecciones_vacio(cliente):
    assert cliente.get("/colecciones").get_json() == {"colecciones": [], "hay_sin": False}


# ---------- eliminar ----------


def test_eliminar_archivo_indexado(biblioteca, entorno):
    assert (entorno / "biologia.txt").exists()
    r = biblioteca.post("/eliminar", json={"archivo": "biologia.txt"})
    assert r.status_code == 200
    assert r.get_json()["exito"] is True

    assert "biologia.txt" not in database.obtener_archivos_indexados()
    assert not (entorno / "biologia.txt").exists()

    # Ya no aparece en búsquedas, pero los otros sí.
    assert biblioteca.get("/buscar?q=fotosintesis").get_json() == []
    assert biblioteca.get("/buscar?q=torta").get_json()[0]["archivo"] == "cocina.txt"
    assert (entorno / "cocina.txt").exists()


def test_eliminar_archivo_con_coleccion(biblioteca, entorno):
    subir(biblioteca, "torta.txt", COCINA, coleccion="recetas")
    assert (entorno / "recetas" / "torta.txt").exists()

    r = biblioteca.post("/eliminar", json={"archivo": "recetas/torta.txt"})
    assert r.get_json()["exito"] is True
    assert not (entorno / "recetas" / "torta.txt").exists()
    assert "recetas/torta.txt" not in database.obtener_archivos_indexados()
    # La colección 'recetas' desaparece de /colecciones
    info = biblioteca.get("/colecciones").get_json()
    assert "recetas" not in info["colecciones"]


def test_eliminar_ultimo_archivo_de_coleccion_la_borra(biblioteca, entorno):
    subir(biblioteca, "unico.txt", COCINA, coleccion="solo")
    assert biblioteca.get("/colecciones").get_json()["colecciones"] == ["solo"]
    biblioteca.post("/eliminar", json={"archivo": "solo/unico.txt"})
    assert biblioteca.get("/colecciones").get_json()["colecciones"] == []


def test_eliminar_inexistente_da_404(biblioteca):
    assert biblioteca.post("/eliminar", json={"archivo": "no_existe.txt"}).status_code == 404


def test_eliminar_sin_archivo_da_400(biblioteca):
    assert biblioteca.post("/eliminar", json={}).status_code == 400
    assert biblioteca.post("/eliminar", json={"archivo": ""}).status_code == 400


def test_eliminar_no_sale_de_la_carpeta(biblioteca, entorno):
    (entorno.parent / "secreto.txt").write_text("x" * 100, encoding="utf-8")
    for malo in ["../secreto.txt", "..\\secreto.txt", "/etc/passwd", "a/../../secreto.txt"]:
        assert biblioteca.post("/eliminar", json={"archivo": malo}).status_code == 400
    assert (entorno.parent / "secreto.txt").exists()


def test_eliminar_solo_acepta_post(biblioteca):
    assert biblioteca.get("/eliminar").status_code == 405


def test_eliminar_archivo_fisico_ausente_limpia_db(biblioteca, entorno):
    # Simula: la DB tiene el archivo pero el .txt ya no está en disco.
    (entorno / "biologia.txt").unlink()
    r = biblioteca.post("/eliminar", json={"archivo": "biologia.txt"})
    assert r.status_code == 200
    assert "biologia.txt" not in database.obtener_archivos_indexados()


# ---------- seguridad ----------


def test_nombre_de_archivo_no_puede_escapar_de_la_carpeta(cliente, entorno):
    subir(cliente, "../../evil.txt", COCINA)
    subir(cliente, "..\\..\\evil2.txt", COCINA)
    assert (entorno / "evil.txt").exists()
    assert (entorno / "evil2.txt").exists()
    assert not (entorno.parent.parent / "evil.txt").exists()
    assert not (entorno.parent / "evil.txt").exists()


def test_coleccion_no_puede_escapar_de_la_carpeta(cliente, entorno):
    subir(cliente, "x.txt", COCINA, coleccion="../../afuera")
    assert (entorno / "afuera" / "x.txt").exists()
    assert not (entorno.parent.parent / "afuera").exists()


def test_limpiar_nombre(cliente):
    import servidor

    assert servidor._limpiar_nombre("a/b/c.pdf") == "c.pdf"
    assert servidor._limpiar_nombre("..\\x.txt") == "x.txt"
    assert servidor._limpiar_nombre("..") == ""
    assert servidor._limpiar_nombre(".") == ""
    assert servidor._limpiar_nombre(None) == ""
    assert servidor._limpiar_nombre("  a.txt ") == "a.txt"


# ---------- descargar / ver / pagina ----------


def test_descargar_y_ver_conservan_la_carpeta(cliente):
    subir(cliente, "torta.txt", COCINA, coleccion="recetas")
    descarga = cliente.get("/descargar/recetas/torta.txt")
    assert descarga.status_code == 200
    assert "attachment" in descarga.headers["Content-Disposition"]
    assert descarga.data.decode("utf-8") == COCINA
    ver = cliente.get("/ver/recetas/torta.txt")
    assert ver.status_code == 200
    assert "attachment" not in ver.headers.get("Content-Disposition", "")


def test_descargar_inexistente_da_404(cliente):
    assert cliente.get("/descargar/no_esta.pdf").status_code == 404
    assert cliente.get("/ver/no_esta.pdf").status_code == 404


def test_descargar_no_sale_de_la_carpeta(cliente, entorno):
    (entorno.parent / "secreto.txt").write_text("secreto", encoding="utf-8")
    assert cliente.get("/descargar/../secreto.txt").status_code == 404


def test_pagina(biblioteca):
    datos = biblioteca.get("/pagina?archivo=biologia.txt&n=1").get_json()
    assert datos["texto"] == BIOLOGIA
    assert datos["anterior"] is None and datos["siguiente"] is None
    assert biblioteca.get("/pagina?archivo=biologia.txt&n=99").status_code == 404
    assert biblioteca.get("/pagina?archivo=biologia.txt&n=abc").status_code == 400


# ---------- info / qr ----------


def test_info_en_modo_local(cliente, monkeypatch):
    import servidor

    monkeypatch.setattr(servidor, "obtener_ip_local", lambda: "192.168.0.25")
    info = cliente.get("/info").get_json()
    assert info["url"].startswith("http://192.168.0.25:")
    assert info["hay_red"] is True
    assert info["es_local"] is True


def test_info_sin_red(cliente, monkeypatch):
    import servidor

    monkeypatch.setattr(servidor, "obtener_ip_local", lambda: "127.0.0.1")
    assert cliente.get("/info").get_json()["hay_red"] is False


def test_qr_sin_segno_da_404(cliente, monkeypatch):
    import servidor

    monkeypatch.setattr(servidor, "segno", None)
    assert cliente.get("/qr.svg").status_code == 404
    assert cliente.get("/info").get_json()["hay_qr"] is False


def test_qr_con_segno_devuelve_svg(cliente):
    pytest.importorskip("segno")
    respuesta = cliente.get("/qr.svg")
    assert respuesta.status_code == 200
    assert respuesta.mimetype == "image/svg+xml"
    assert b"<svg" in respuesta.data


# ---------- apagar ----------


@pytest.fixture
def timer_falso(cliente, monkeypatch):
    """Evita que /apagar llame de verdad a os._exit (mataria a pytest)."""
    import servidor

    arrancados = []

    class TimerFalso:
        def __init__(self, segundos, funcion):
            self.funcion = funcion

        def start(self):
            arrancados.append(self)

    monkeypatch.setattr(servidor, "threading", types.SimpleNamespace(Timer=TimerFalso))
    return arrancados


def test_apagar_local_con_encabezado(cliente, timer_falso):
    respuesta = cliente.post("/apagar", headers={"X-Biblioteca": "1"})
    assert respuesta.status_code == 200
    assert len(timer_falso) == 1


def test_apagar_sin_encabezado_es_rechazado(cliente, timer_falso):
    assert cliente.post("/apagar").status_code == 403
    assert timer_falso == []


def test_apagar_desde_otro_equipo_es_rechazado(cliente, timer_falso):
    respuesta = cliente.post(
        "/apagar", headers={"X-Biblioteca": "1"}, environ_base={"REMOTE_ADDR": "8.8.8.8"}
    )
    assert respuesta.status_code == 403
    assert timer_falso == []


def test_apagar_no_acepta_get(cliente, timer_falso):
    assert cliente.get("/apagar").status_code == 405
    assert timer_falso == []
