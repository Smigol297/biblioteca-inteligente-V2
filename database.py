import sqlite3

from config import DB_PATH


def obtener_conexion():
    """Crea una conexión limpia con la base de datos."""
    return sqlite3.connect(DB_PATH)


def inicializar_db():
    """Crea la tabla necesaria si no existe en el disco rígido."""
    with obtener_conexion() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS paginas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                archivo TEXT,
                pagina INTEGER,
                texto TEXT
            )
        """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_archivo_pagina ON paginas(archivo, pagina)")
        conn.commit()


def obtener_archivos_indexados():
    """Devuelve un conjunto con los nombres de archivos ya procesados."""
    with obtener_conexion() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT archivo FROM paginas")
        return {row[0] for row in cursor.fetchall()}


def obtener_todas_las_paginas():
    """Recupera todas las páginas guardadas para alimentar al buscador."""
    with obtener_conexion() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, archivo, pagina, texto FROM paginas")
        return cursor.fetchall()


def reemplazar_archivo(archivo, paginas):
    """Borra las páginas previas de `archivo` e inserta las nuevas, todo en una transacción.
    `paginas` es una lista de tuplas (num_pagina, texto)."""
    conn = obtener_conexion()
    try:
        with conn:  # commit al salir bien, rollback si hay excepción
            conn.execute("DELETE FROM paginas WHERE archivo = ?", (archivo,))
            conn.executemany(
                "INSERT INTO paginas (archivo, pagina, texto) VALUES (?, ?, ?)",
                [(archivo, n, t) for n, t in paginas],
            )
    finally:
        conn.close()  # `with conn` no cierra la conexión, por eso el finally


def obtener_pagina(archivo, pagina):
    """Devuelve el texto de una página y las páginas vecinas que existen en la base."""
    conn = obtener_conexion()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT texto FROM paginas WHERE archivo = ? AND pagina = ? LIMIT 1", (archivo, pagina)
        )
        fila = cur.fetchone()
        if fila is None:
            return None
        cur.execute(
            "SELECT MAX(pagina) FROM paginas WHERE archivo = ? AND pagina < ?", (archivo, pagina)
        )
        anterior = cur.fetchone()[0]
        cur.execute(
            "SELECT MIN(pagina) FROM paginas WHERE archivo = ? AND pagina > ?", (archivo, pagina)
        )
        siguiente = cur.fetchone()[0]
        cur.execute("SELECT MAX(pagina) FROM paginas WHERE archivo = ?", (archivo,))
        ultima = cur.fetchone()[0]
        return {
            "archivo": archivo,
            "pagina": pagina,
            "texto": fila[0],
            "anterior": anterior,
            "siguiente": siguiente,
            "ultima": ultima,
        }
    finally:
        conn.close()
