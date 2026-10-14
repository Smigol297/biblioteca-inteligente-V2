from docx import Document

import database
import indexador_service

PARRAFO = "Este es un parrafo con texto suficiente para superar el minimo"


def test_leer_texto_plano_parte_en_paginas_de_30_lineas(tmp_path):
    ruta = tmp_path / "libro.txt"
    ruta.write_text("\n".join("linea %d" % i for i in range(35)), encoding="utf-8")
    paginas = indexador_service.leer_texto_plano(str(ruta))
    assert [n for n, _ in paginas] == [1, 2]


def test_leer_texto_plano_tolera_bytes_invalidos(tmp_path):
    ruta = tmp_path / "roto.txt"
    ruta.write_bytes(b"hola \xff\xfe mundo")
    paginas = indexador_service.leer_texto_plano(str(ruta))
    assert "hola" in paginas[0][1]


def test_leer_docx_parte_en_paginas_de_15_parrafos(tmp_path):
    ruta = tmp_path / "doc.docx"
    doc = Document()
    for i in range(16):
        doc.add_paragraph("%s %d" % (PARRAFO, i))
    doc.save(str(ruta))
    paginas = indexador_service.leer_docx(str(ruta))
    assert [n for n, _ in paginas] == [1, 2]


def test_leer_pdf_extrae_texto_por_pagina(tmp_path, crear_pdf):
    ruta = crear_pdf(tmp_path / "libro.pdf", ["Hola primera pagina", "Segunda pagina aca"])
    paginas = indexador_service.leer_pdf(str(ruta))
    assert [n for n, _ in paginas] == [1, 2]
    assert "primera" in paginas[0][1]


def test_procesar_txt_guarda_en_la_base(entorno):
    ruta = entorno / "notas.txt"
    ruta.write_text(PARRAFO + " numero uno", encoding="utf-8")
    exito, resultado = indexador_service.procesar_un_archivo(str(ruta), "notas.txt")
    assert (exito, resultado) == (True, 1)
    assert database.obtener_archivos_indexados() == {"notas.txt"}


def test_procesar_normaliza_espacios(entorno):
    ruta = entorno / "notas.txt"
    ruta.write_text(
        "hola     mundo\t\ttexto largo para pasar el minimo permitido", encoding="utf-8"
    )
    indexador_service.procesar_un_archivo(str(ruta), "notas.txt")
    texto = database.obtener_todas_las_paginas()[0][3]
    assert "  " not in texto and "\t" not in texto


def test_procesar_usa_la_clave_con_coleccion(entorno):
    ruta = entorno / "notas.txt"
    ruta.write_text(PARRAFO, encoding="utf-8")
    indexador_service.procesar_un_archivo(str(ruta), "mi_coleccion/notas.txt")
    assert database.obtener_archivos_indexados() == {"mi_coleccion/notas.txt"}


def test_procesar_formato_no_soportado(entorno):
    ruta = entorno / "datos.xyz"
    ruta.write_text(PARRAFO, encoding="utf-8")
    exito, mensaje = indexador_service.procesar_un_archivo(str(ruta), "datos.xyz")
    assert exito is False
    assert "no soportado" in mensaje


def test_procesar_texto_demasiado_corto(entorno):
    ruta = entorno / "corto.txt"
    ruta.write_text("muy corto", encoding="utf-8")
    exito, mensaje = indexador_service.procesar_un_archivo(str(ruta), "corto.txt")
    assert exito is False
    assert "No se pudo extraer texto" in mensaje
    assert database.obtener_archivos_indexados() == set()


def test_procesar_archivo_inexistente_no_explota(entorno):
    exito, _ = indexador_service.procesar_un_archivo(str(entorno / "no_esta.txt"), "no_esta.txt")
    assert exito is False
