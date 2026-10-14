import database


def test_reemplazar_archivo_no_duplica(entorno):
    database.reemplazar_archivo("libro.txt", [(1, "uno"), (2, "dos")])
    database.reemplazar_archivo("libro.txt", [(1, "uno nuevo"), (2, "dos nuevo")])
    filas = database.obtener_todas_las_paginas()
    assert len(filas) == 2
    assert {f[3] for f in filas} == {"uno nuevo", "dos nuevo"}


def test_reemplazar_archivo_no_toca_otros_libros(entorno):
    database.reemplazar_archivo("a.txt", [(1, "aaa")])
    database.reemplazar_archivo("b.txt", [(1, "bbb")])
    database.reemplazar_archivo("a.txt", [(1, "aaa 2")])
    assert database.obtener_archivos_indexados() == {"a.txt", "b.txt"}


def test_obtener_pagina_devuelve_vecinas_existentes(entorno):
    database.reemplazar_archivo("libro.txt", [(1, "p1"), (3, "p3"), (7, "p7")])
    datos = database.obtener_pagina("libro.txt", 3)
    assert datos["texto"] == "p3"
    assert datos["anterior"] == 1
    assert datos["siguiente"] == 7
    assert datos["ultima"] == 7


def test_obtener_pagina_en_los_extremos(entorno):
    database.reemplazar_archivo("libro.txt", [(1, "p1"), (2, "p2")])
    assert database.obtener_pagina("libro.txt", 1)["anterior"] is None
    assert database.obtener_pagina("libro.txt", 2)["siguiente"] is None


def test_obtener_pagina_inexistente_devuelve_none(entorno):
    database.reemplazar_archivo("libro.txt", [(1, "p1")])
    assert database.obtener_pagina("libro.txt", 99) is None
    assert database.obtener_pagina("otro.txt", 1) is None


def test_inicializar_db_es_idempotente(entorno):
    database.inicializar_db()
    database.inicializar_db()
    assert database.obtener_todas_las_paginas() == []
