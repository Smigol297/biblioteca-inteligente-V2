# -*- coding: utf-8 -*-
"""Fixtures compartidos por todos los tests."""
import numpy as np
import pytest

import config
import database

# ---------------------------------------------------------------------------
# Aislamiento: base SQLite y carpeta mis_pdfs temporales por test.
# ---------------------------------------------------------------------------


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """Carpeta mis_pdfs/ y base SQLite temporales, aisladas por test.

    Tambien patchea BASE_DIR (para que embeddings.npy no caiga en el repo) y
    MODELOS_DIR (por las dudas de que algo consulte la ruta del modelo ONNX).
    """
    base = tmp_path / "biblioteca"
    base.mkdir()
    pdfs = base / "mis_pdfs"
    pdfs.mkdir()
    modelos = base / "model_cache"
    modelos.mkdir()
    monkeypatch.setattr(config, "BASE_DIR", str(base))
    monkeypatch.setattr(config, "CARPETA_PDFS", str(pdfs))
    monkeypatch.setattr(config, "MODELOS_DIR", str(modelos))
    monkeypatch.setattr(database, "DB_PATH", str(base / "biblioteca.db"))
    database.inicializar_db()
    return pdfs


# ---------------------------------------------------------------------------
# Fake de fastembed.TextEmbedding: sin descarga, sin internet, sin ONNX.
# ---------------------------------------------------------------------------


class _FakeEmbedding:
    """Reemplazo determinista de TextEmbedding para tests.

    Es una bolsa de palabras hasheada: dos textos con palabras en comun
    tienen producto punto > 0, lo cual alcanza para verificar que el denso
    aporta algo. No descarga nada ni toca internet.

    Guarda los textos que recibe en `textos_recibidos` (variable de clase)
    para que los tests puedan verificar los prefijos "query:" / "passage:".
    """

    DIM = 384
    textos_recibidos = []

    def __init__(self, *args, **kwargs):
        # Ignoramos model_name, cache_dir y demas: al fake le da igual.
        pass

    def embed(self, textos):
        for t in textos:
            _FakeEmbedding.textos_recibidos.append(t)
            v = np.zeros(self.DIM, dtype=np.float32)
            for palabra in t.lower().split():
                idx = abs(hash(palabra)) % self.DIM
                v[idx] += 1.0
            n = float(np.linalg.norm(v))
            if n > 0:
                v = v / n
            yield v


@pytest.fixture(autouse=True)
def sin_modelo_real(monkeypatch):
    """Evita descargar el modelo ONNX (220 MB) en cada corrida de tests.

    Reemplaza TextEmbedding por el fake. Autouse: aplica a todo test que
    instancie un BuscadorHibrido, sin que el test tenga que pedirlo.
    """
    import buscador_service

    monkeypatch.setattr(buscador_service, "TextEmbedding", _FakeEmbedding)
    _FakeEmbedding.textos_recibidos = []


@pytest.fixture
def fake_embedding():
    """Expone la clase fake para que los tests inspeccionen sus capturas."""
    return _FakeEmbedding


# ---------------------------------------------------------------------------
# Constructor de BuscadorHibrido listo para usar.
# ---------------------------------------------------------------------------


@pytest.fixture
def crear_buscador(entorno):
    """Devuelve una funcion ({'libro.txt': ['pag1', 'pag2']}) -> BuscadorHibrido entrenado.

    Reemplaza a la vieja que devolvia BuscadorLocal. La firma es la misma,
    asi que los tests que ya la usan siguen andando sin cambios.
    """
    from buscador_service import BuscadorHibrido

    def _crear(libros):
        for nombre, paginas in libros.items():
            database.reemplazar_archivo(nombre, list(enumerate(paginas, start=1)))
        b = BuscadorHibrido()
        assert b.entrenar_modelo() is True
        return b

    return _crear


# ---------------------------------------------------------------------------
# Cliente Flask aislado.
# ---------------------------------------------------------------------------


@pytest.fixture
def cliente(entorno):
    """Cliente de pruebas de Flask con base y carpetas aisladas.

    Importa servidor dentro del fixture (no al tope del modulo) porque
    servidor.py inicializa la base y entrena el buscador al importarse.
    """
    import servidor

    servidor.app.config["TESTING"] = True
    with servidor.app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Generador de PDFs minimos (una pagina por texto, solo ASCII).
# ---------------------------------------------------------------------------


@pytest.fixture
def crear_pdf():
    """Devuelve una funcion (ruta, [texto1, texto2, ...]) -> ruta del PDF."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    def _crear(ruta, paginas):
        writer = PdfWriter()
        for texto in paginas:
            pagina = writer.add_blank_page(width=612, height=792)
            # Fuente Helvetica con encoding WinAnsi: sin esto, pypdf lee los
            # bytes del stream pero no sabe mapearlos a caracteres.
            font = DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                    NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
                }
            )
            font_ref = writer._add_object(font)
            pagina[NameObject("/Resources")] = DictionaryObject(
                {
                    NameObject("/Font"): DictionaryObject(
                        {
                            NameObject("/F1"): font_ref,
                        }
                    ),
                }
            )
            contenido = "BT /F1 12 Tf 72 720 Td (%s) Tj ET" % texto
            stream = DecodedStreamObject()
            stream.set_data(contenido.encode("latin-1", errors="replace"))
            pagina[NameObject("/Contents")] = writer._add_object(stream)
        with open(str(ruta), "wb") as f:
            writer.write(f)
        return ruta

    return _crear
