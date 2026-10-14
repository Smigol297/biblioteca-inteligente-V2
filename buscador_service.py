import re
from bisect import bisect_left

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from translate import Translator

import database


def _coleccion_de(archivo):
    """La colección es la carpeta: 'recetas/torta.pdf' -> 'recetas'. Sin carpeta -> ''."""
    return archivo.split("/")[0] if "/" in archivo else ""


def _fragmento(texto, terminos, ancho=300):
    """Devuelve [[trozo, resaltado], ...]: la ventana de ~ancho caracteres con más términos distintos."""
    spans = []
    if terminos:
        alternativa = "|".join(re.escape(t) for t in sorted(terminos, key=len, reverse=True))
        patron = re.compile(r"\b(?:" + alternativa + r")\b", re.IGNORECASE)
        for m in patron.finditer(texto):
            spans.append((m.start(), m.end(), m.group(0).lower()))

    if not spans:
        corte = texto[:ancho]
        return [[corte + ("…" if len(texto) > ancho else ""), 0]]

    # Elegir el comienzo de ventana que encierre más términos distintos (y después más apariciones)
    starts = [s[0] for s in spans]
    mejor_ini, mejor_pun = 0, (-1, -1)
    for s_ini, _, _ in spans[:50]:
        ini = max(0, s_ini - 60)
        j = bisect_left(starts, ini)
        dentro = []
        while j < len(spans) and spans[j][0] < ini + ancho:
            if spans[j][1] <= ini + ancho:
                dentro.append(spans[j])
            j += 1
        pun = (len(set(s[2] for s in dentro)), len(dentro))
        if pun > mejor_pun:
            mejor_ini, mejor_pun = ini, pun

    # Ajustar los bordes a límites de palabra
    ini = mejor_ini
    if ini > 0:
        sp = texto.find(" ", ini, ini + 20)
        if sp != -1:
            ini = sp + 1
    fin = min(len(texto), mejor_ini + ancho)
    if fin < len(texto):
        sp = texto.rfind(" ", max(ini, fin - 20), fin)
        if sp != -1:
            fin = sp

    trozos = []
    pos = ini
    for s_ini, s_fin, _ in spans:
        if s_ini < pos or s_fin > fin:
            continue
        if s_ini > pos:
            trozos.append([texto[pos:s_ini], 0])
        trozos.append([texto[s_ini:s_fin], 1])
        pos = s_fin
    if pos < fin:
        trozos.append([texto[pos:fin], 0])
    if ini > 0:
        trozos.insert(0, ["… ", 0])
    if fin < len(texto):
        trozos.append([" …", 0])
    return trozos


class BuscadorLocal:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(lowercase=True)
        self.matrix_tfidf = None
        self.paginas_data = []
        self.colecciones = None

    def entrenar_modelo(self):
        """Carga los datos desde el disco y entrena el motor matemático local."""
        filas = database.obtener_todas_las_paginas()
        if not filas:
            return False

        self.paginas_data = filas
        self.colecciones = np.array([_coleccion_de(r[1]) for r in filas])
        textos = [r[3] for r in filas]

        self.matrix_tfidf = self.vectorizer.fit_transform(textos)
        return True

    def _traducir_consulta(self, query):
        """Intenta traducir la consulta del español al inglés de forma segura."""
        try:
            translator = Translator(from_lang="es", to_lang="en")
            traduccion = translator.translate(query)
            if traduccion and traduccion.lower() != query.lower():
                return f"{query} {traduccion}"
        except Exception as e:
            print(f"Error en traducción: {e}")
        return query

    def _terminos_resaltables(self, query_expandida):
        """Palabras de la consulta (es + en) que vale la pena resaltar.
        Se descartan las muy comunes ('de', 'el') para no resaltar todo."""
        vocab = self.vectorizer.vocabulary_
        idf = self.vectorizer.idf_
        terminos = set(re.findall(r"\w\w+", query_expandida.lower()))
        raros = set(t for t in terminos if t in vocab and idf[vocab[t]] > 1.7)
        return raros or set(t for t in terminos if t in vocab)

    def buscar(
        self, query, top_k=100, umbral_abs=0.03, umbral_rel=0.3, agrupar=False, coleccion=None
    ):
        """Búsqueda bilingüe: devuelve solo coincidencias fuertes.
        agrupar=True deja solo la mejor página de cada libro."""
        if self.matrix_tfidf is None:
            return []

        query_expandida = self._traducir_consulta(query)
        print(f" Buscando: {query_expandida}")

        vector_pregunta = self.vectorizer.transform([query_expandida])
        similitudes = cosine_similarity(self.matrix_tfidf, vector_pregunta).flatten()
        if coleccion is not None:  # None = todas; '' = solo los que no tienen colección
            similitudes = np.where(self.colecciones == coleccion, similitudes, 0.0)
        mejor = similitudes.max()
        if mejor <= 0:
            return []
        minimo = max(umbral_abs, mejor * umbral_rel)
        terminos = self._terminos_resaltables(query_expandida)

        resultados = []
        vistos = set()
        for idx in np.argsort(similitudes)[::-1]:
            if similitudes[idx] < minimo:
                break  # están ordenados: los siguientes son todavía peores
            id_db, archivo, pagina, texto = self.paginas_data[idx]
            if agrupar:
                if archivo in vistos:
                    continue  # ya tenemos la mejor página de este libro
                vistos.add(archivo)
            fragmento = _fragmento(texto, terminos)
            resultados.append(
                {
                    "archivo": archivo,
                    "pagina": int(pagina),
                    "texto": "".join(t for t, _ in fragmento),
                    "fragmento": fragmento,
                    "score": round(float(similitudes[idx]), 3),
                }
            )
            if len(resultados) >= top_k:
                break
        return resultados
