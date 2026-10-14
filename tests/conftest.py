"""Fixtures compartidas. Ningun test toca internet, ni la base ni la carpeta reales."""

import pytest

import buscador_service
import config
import database


@pytest.fixture(autouse=True)
def sin_internet(monkeypatch):
    """La traduccion usa internet: en los tests devuelve la consulta tal cual."""
    monkeypatch.setattr(
        buscador_service.BuscadorLocal, "_traducir_consulta", lambda self, query: query
    )


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """Carpeta mis_pdfs y base SQLite temporales (se borran solas al terminar el test)."""
    carpeta = tmp_path / "mis_pdfs"
    carpeta.mkdir()
    monkeypatch.setattr(config, "CARPETA_PDFS", str(carpeta))
    # database.py hizo `from config import DB_PATH`, por eso se parchea en database
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "biblioteca.db"))
    database.inicializar_db()
    return carpeta


@pytest.fixture
def crear_buscador(entorno):
    """Devuelve una funcion: crear_buscador({'libro.txt': ['pag 1', 'pag 2']}) -> buscador entrenado."""

    def _crear(libros):
        for archivo, paginas in libros.items():
            database.reemplazar_archivo(archivo, list(enumerate(paginas, start=1)))
        buscador = buscador_service.BuscadorLocal()
        assert buscador.entrenar_modelo()
        return buscador

    return _crear


@pytest.fixture
def cliente(entorno, monkeypatch):
    """Cliente de pruebas de Flask con buscador y carpetas aislados.

    servidor.py inicializa la base al importarse, por eso se importa recien aca,
    con las rutas temporales ya parchadas."""
    import servidor

    monkeypatch.setattr(servidor, "buscador", buscador_service.BuscadorLocal())
    servidor.app.config["TESTING"] = True
    return servidor.app.test_client()


@pytest.fixture
def crear_pdf():
    """Devuelve una funcion que escribe un PDF minimo (una pagina por texto, solo ASCII)."""

    def _crear(ruta, textos):
        n = len(textos)
        kids = " ".join("%d 0 R" % (4 + 2 * i) for i in range(n))
        objetos = [
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, n),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
        for i, texto in enumerate(textos):
            contenido = "BT /F1 12 Tf 72 720 Td (%s) Tj ET" % texto
            objetos.append(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents %d 0 R "
                "/Resources << /Font << /F1 3 0 R >> >> >>" % (5 + 2 * i)
            )
            objetos.append("<< /Length %d >>\nstream\n%s\nendstream" % (len(contenido), contenido))

        salida = b"%PDF-1.4\n"
        offsets = []
        for num, cuerpo in enumerate(objetos, start=1):
            offsets.append(len(salida))
            salida += ("%d 0 obj\n%s\nendobj\n" % (num, cuerpo)).encode("latin-1")
        inicio_xref = len(salida)
        salida += ("xref\n0 %d\n0000000000 65535 f \n" % (len(objetos) + 1)).encode()
        for off in offsets:
            salida += ("%010d 00000 n \n" % off).encode()
        salida += (
            "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objetos) + 1, inicio_xref)
        ).encode()
        with open(str(ruta), "wb") as f:
            f.write(salida)
        return ruta

    return _crear
