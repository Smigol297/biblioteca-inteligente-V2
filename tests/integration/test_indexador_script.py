import database
import indexador


def test_ejecutar_indexacion_indexa_los_archivos_nuevos(entorno, capsys):
    (entorno / "uno.txt").write_text(
        "Texto de prueba lo bastante largo para indexarse", encoding="utf-8"
    )
    (entorno / "ignorado.jpg").write_bytes(b"\xff\xd8\xff")

    indexador.ejecutar_indexacion()
    assert database.obtener_archivos_indexados() == {"uno.txt"}

    capsys.readouterr()
    indexador.ejecutar_indexacion()  # segunda corrida: no hay nada nuevo
    assert "No se encontraron" in capsys.readouterr().out
