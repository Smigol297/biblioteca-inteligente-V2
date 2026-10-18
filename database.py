import sqlite3

from config import DB_PATH


def obtener_conexion():
    """Crea una conexión limpia con la base de datos."""
    return sqlite3.connect(DB_PATH)


def inicializar_db():
    """Crea las tablas necesarias si no existen en el disco rígido."""
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
        # Embeddings por archivo: una fila por archivo, con el modelo que la
        # genero. Si cambia MODELO_ID, las filas viejas quedan invisibles
        # (se filtran por modelo_id) y se re-embeben al primer entrenar.
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                archivo     TEXT PRIMARY KEY,
                modelo_id   TEXT NOT NULL,
                hash        TEXT NOT NULL,
                dim         INTEGER NOT NULL,
                n_paginas   INTEGER NOT NULL,
                data        BLOB NOT NULL,
                actualizado TEXT NOT NULL
            )
        """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_emb_modelo ON embeddings(modelo_id)")
        conn.commit()


def obtener_archivos_indexados():
    """Devuelve un conjunto con los nombres de archivos ya procesados."""
    with obtener_conexion() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT archivo FROM paginas")
        return {row[0] for row in cursor.fetchall()}


def obtener_todas_las_paginas():
    """Recupera todas las páginas guardadas para alimentar al buscador.

    ORDER BY id explícito (antes era implícito). Los embeddings por archivo
    se ensamblan respetando este orden, así que tiene que ser determinístico.
    """
    with obtener_conexion() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, archivo, pagina, texto FROM paginas ORDER BY id")
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


# ---------------------------------------------------------------------------
# Embeddings por archivo (indexado incremental)
# ---------------------------------------------------------------------------


def obtener_embeddings(modelo_id):
    """Devuelve {archivo: (hash, n_paginas, dim, blob)} para un modelo_id dado.

    Las filas con otro modelo_id se ignoran: al bumpear MODELO_ID todo se
    re-embebe automáticamente en el siguiente entrenar_modelo().
    """
    with obtener_conexion() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT archivo, hash, n_paginas, dim, data FROM embeddings WHERE modelo_id = ?",
            (modelo_id,),
        )
        return {row[0]: (row[1], row[2], row[3], row[4]) for row in cur.fetchall()}


def guardar_embeddings_bulk(items, modelo_id):
    """items: iterable de (archivo, hash, emb_array). Upsert por archivo.

    Todo en una transacción: si se corta a mitad, no queda la tabla
    a medio actualizar. INSERT OR REPLACE porque archivo es PRIMARY KEY.
    """
    import datetime as _dt

    import numpy as _np

    ahora = _dt.datetime.now().isoformat(timespec="seconds")
    conn = obtener_conexion()
    try:
        with conn:
            for archivo, hash_, emb in items:
                emb = _np.ascontiguousarray(emb, dtype=_np.float32)
                n, dim = emb.shape
                conn.execute(
                    """INSERT OR REPLACE INTO embeddings
                       (archivo, modelo_id, hash, dim, n_paginas, data, actualizado)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (archivo, modelo_id, hash_, int(dim), int(n), emb.tobytes(), ahora),
                )
    finally:
        conn.close()


def borrar_embeddings_archivos(archivos):
    """Borra los embeddings de una lista de archivos (huérfanos)."""
    if not archivos:
        return
    conn = obtener_conexion()
    try:
        with conn:
            conn.executemany("DELETE FROM embeddings WHERE archivo = ?", [(a,) for a in archivos])
    finally:
        conn.close()

def borrar_archivo(archivo):
    """Borra las páginas y el embedding de un archivo en una sola transacción.

    Devuelve True si había filas en `paginas` (es decir, si el archivo estaba
    indexado), False si no existía. La transacción es todo-o-nada: si algo
    falla, tanto `paginas` como `embeddings` quedan como estaban.
    """
    conn = obtener_conexion()
    try:
        with conn:
            cur = conn.execute("SELECT COUNT(1) FROM paginas WHERE archivo = ?", (archivo,))
            if cur.fetchone()[0] == 0:
                return False
            conn.execute("DELETE FROM paginas WHERE archivo = ?", (archivo,))
            conn.execute("DELETE FROM embeddings WHERE archivo = ?", (archivo,))
        return True
    finally:
        conn.close()