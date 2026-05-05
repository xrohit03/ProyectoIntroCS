"""
CS1000 - Introducción a las Ciencias de la Computación
Proyecto 1 (P1) - 2026-1
Profesor: Ian Paul Brossard

Análisis de Pokémon mediante TF-IDF y K-Means clustering
Dataset: smogon.csv - columna "moves"
"""
                                                                
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# ============================================================
# CARGA Y PREPROCESAMIENTO DEL DATASET
# ============================================================

print("=" * 70)
print("CARGANDO DATASET")
print("=" * 70)

df = pd.read_csv('smogon.csv', on_bad_lines='skip')
print(f"Dataset cargado: {df.shape[0]} Pokémon, {df.shape[1]} columnas")
print(f"Columnas: {df.columns.tolist()}")
print(f"Valores nulos en 'moves': {df['moves'].isna().sum()}")

# Rellenar valores nulos con cadena vacía
df['moves'] = df['moves'].fillna('')

# ============================================================
# LOS 18 TIPOS DE POKÉMON
# ============================================================

TIPOS_POKEMON = [
    'normal', 'fire', 'water', 'electric', 'grass', 'ice',
    'fighting', 'poison', 'ground', 'flying', 'psychic', 'bug',
    'rock', 'ghost', 'dragon', 'dark', 'steel', 'fairy'
]

# ============================================================
# PREGUNTA 1: AGRUPAMIENTO MEDIANTE TF-IDF CON N-GRAMAS
# ============================================================

print("\n" + "=" * 70)
print("PREGUNTA 1: AGRUPAMIENTO MEDIANTE TF-IDF CON N-GRAMAS")
print("=" * 70)

# --- 1.1 Crear TfidfVectorizer con bigramas (n_gram_range=(1,2)) ---
# Usamos unigramas + bigramas para capturar frases como "fire punch", "water gun", etc.
print("\n[1.1] Generando matriz TF-IDF con unigramas y bigramas...")

tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),       # unigramas y bigramas
    max_features=5000,         # limitamos vocabulario para evitar ruido
    min_df=2,                  # el n-grama debe aparecer en al menos 2 documentos
    sublinear_tf=True,         # aplica log(tf) para suavizar frecuencias altas
    token_pattern=r'[a-zA-Z]{2,}'  # solo tokens alfabéticos de al menos 2 letras
)

moves_corpus = df['moves'].tolist()
tfidf_matrix = tfidf_vectorizer.fit_transform(moves_corpus)

vocabulario = tfidf_vectorizer.get_feature_names_out()

print(f"\n[1.2] Número total de columnas (elementos del vocabulario): {len(vocabulario)}")

print(f"\n[1.3] Primeros 100 elementos del vocabulario:")
for i, term in enumerate(vocabulario[:100]):
    print(f"  {i+1:3d}. {term}")
print(f"  ... (total: {len(vocabulario)} elementos)")

# --- 1.4 Generar DataFrame con la matriz TF-IDF ---
print("\n[1.4] Generando DataFrame con la matriz TF-IDF...")
df_tfidf = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vocabulario,
    index=df['Pokemon']
)
print("Imprimiendo matriz TF-IDF (print):")
print(df_tfidf)
print(f"\nDimensiones del DataFrame TF-IDF: {df_tfidf.shape}")

# --- 1.5 Clustering con K-Means ---
# Elegimos 18 clusters (uno por tipo de Pokémon)
N_CLUSTERS_P1 = 18
print(f"\n[1.5] Aplicando K-Means con {N_CLUSTERS_P1} clusters...")

# Normalizamos para mejor clustering con TF-IDF
tfidf_norm = normalize(tfidf_matrix)

kmeans_p1 = KMeans(
    n_clusters=N_CLUSTERS_P1,
    random_state=42,
    n_init=20,
    max_iter=500
)
kmeans_p1.fit(tfidf_norm)
labels_p1 = kmeans_p1.labels_

df['cluster_p1'] = labels_p1

# --- 1.6 Guardar CSV con Pokémon y cluster ---
resultado_p1 = df[['Pokemon', 'cluster_p1']].copy()
resultado_p1.to_csv('resultado_pregunta1.csv', index=False)
print("\n[1.6] Archivo 'resultado_pregunta1.csv' guardado.")

# --- 1.7 Mostrar contenido de cada cluster ---
print("\n[1.7] Contenido de cada cluster (Pregunta 1):")
for c in range(N_CLUSTERS_P1):
    pokemon_cluster = df[df['cluster_p1'] == c]['Pokemon'].tolist()
    print(f"\n  Cluster {c} ({len(pokemon_cluster)} Pokémon):")
    print(f"  {', '.join(pokemon_cluster[:20])}", end='')
    if len(pokemon_cluster) > 20:
        print(f" ... (+{len(pokemon_cluster)-20} más)")
    else:
        print()

# --- Palabras más importantes por cluster (para interpretación) ---
print("\n[1.7b] Top 10 términos TF-IDF más relevantes por cluster (P1):")
order_centroids = kmeans_p1.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vectorizer.get_feature_names_out()

for c in range(N_CLUSTERS_P1):
    top_terms = [terms[ind] for ind in order_centroids[c, :10]]
    print(f"  Cluster {c}: {', '.join(top_terms)}")


# ============================================================
# PREGUNTA 2: VOCABULARIO CONTROLADO (18 TIPOS)
# ============================================================

print("\n" + "=" * 70)
print("PREGUNTA 2: AGRUPAMIENTO CON VOCABULARIO CONTROLADO (18 TIPOS)")
print("=" * 70)

# --- 2.1 Preprocesar: agregar espacios alrededor de cada tipo ---
print("\n[2.1] Preprocesando texto: agregando espacios alrededor de cada tipo...")

def agregar_espacios_tipos(texto, tipos):
    """
    Agrega un espacio a la izquierda y derecha de cada tipo para
    que TfidfVectorizer pueda encontrarlos correctamente.
    """
    texto = str(texto).lower()
    for tipo in tipos:
        # Reemplaza el tipo sin importar si está pegado a otras letras
        # Usa regex para insertar espacio antes y después
        texto = re.sub(r'(?<![a-z])' + tipo + r'(?![a-z])', f' {tipo} ', texto)
        # También maneja casos como "FirePunch" → "fire punch"
        texto = re.sub(tipo.capitalize(), f' {tipo} ', texto)
    return texto

df['moves_procesado'] = df['moves'].apply(
    lambda x: agregar_espacios_tipos(x, TIPOS_POKEMON)
)

# Muestra ejemplo de preprocesamiento
print("\n  Ejemplo de preprocesamiento (Abomasnow, primeros 300 chars):")
print(f"  ORIGINAL: {df['moves'].iloc[0][:200]}")
print(f"  PROCESADO: {df['moves_procesado'].iloc[0][:200]}")

# --- 2.2 TfidfVectorizer con vocabulario controlado ---
print("\n[2.2] Generando TF-IDF con vocabulario controlado (18 tipos)...")

tfidf_vocab_controlado = TfidfVectorizer(
    vocabulary=TIPOS_POKEMON,   # solo busca estos 18 términos
    token_pattern=r'[a-zA-Z]+'
)

moves_procesado_corpus = df['moves_procesado'].tolist()
tfidf_matrix_p2 = tfidf_vocab_controlado.fit_transform(moves_procesado_corpus)

print(f"\n  Vocabulario controlado ({len(TIPOS_POKEMON)} tipos):")
print(f"  {TIPOS_POKEMON}")
print(f"\n  Dimensiones de la matriz TF-IDF (vocabulario controlado): {tfidf_matrix_p2.shape}")

# --- 2.3 DataFrame con la matriz vocabulario controlado ---
df_tfidf_p2 = pd.DataFrame(
    tfidf_matrix_p2.toarray(),
    columns=TIPOS_POKEMON,
    index=df['Pokemon']
)
print("\n  Primeras 10 filas de la matriz TF-IDF (vocabulario controlado):")
print(df_tfidf_p2.head(10))

# --- 2.4 Clustering con K-Means ---
N_CLUSTERS_P2 = 18
print(f"\n[2.4] Aplicando K-Means con {N_CLUSTERS_P2} clusters...")

tfidf_norm_p2 = normalize(tfidf_matrix_p2)

kmeans_p2 = KMeans(
    n_clusters=N_CLUSTERS_P2,
    random_state=42,
    n_init=50,
    max_iter=500
)
kmeans_p2.fit(tfidf_norm_p2)
labels_p2 = kmeans_p2.labels_

df['cluster_p2'] = labels_p2

# --- 2.5 Guardar CSV ---
resultado_p2 = df[['Pokemon', 'cluster_p2']].copy()
resultado_p2.to_csv('resultado_pregunta2.csv', index=False)
print("\n[2.5] Archivo 'resultado_pregunta2.csv' guardado.")

# --- 2.6 Mostrar clusters y tipos dominantes ---
print("\n[2.6] Clusters (Pregunta 2) con tipos dominantes:")
for c in range(N_CLUSTERS_P2):
    idx = df[df['cluster_p2'] == c].index
    pokemon_cluster = df[df['cluster_p2'] == c]['Pokemon'].tolist()
    
    # Promedio de puntajes TF-IDF por tipo en este cluster
    puntajes = df_tfidf_p2.iloc[idx].mean()
    top_tipos = puntajes[puntajes > 0].sort_values(ascending=False).head(3)
    tipos_str = ', '.join([f"{t}={v:.3f}" for t, v in top_tipos.items()])
    
    print(f"\n  Cluster {c} ({len(pokemon_cluster)} Pokémon) → Tipos: {tipos_str}")
    print(f"  Pokémon: {', '.join(pokemon_cluster[:15])}", end='')
    if len(pokemon_cluster) > 15:
        print(f" ... (+{len(pokemon_cluster)-15} más)")
    else:
        print()


# ============================================================
# PREGUNTA 3: VERIFICACIÓN POR TIPOS
# ============================================================

print("\n" + "=" * 70)
print("PREGUNTA 3: VERIFICACIÓN DE AGRUPAMIENTO POR TIPOS")
print("=" * 70)

# --- 3.1 Contar ocurrencias de cada tipo en la columna moves ---
print("\n[3.1] Contando ocurrencias de cada tipo en la columna 'moves'...")

for tipo in TIPOS_POKEMON:
    df[f'count_{tipo}'] = df['moves_procesado'].apply(
        lambda x: len(re.findall(r'\b' + tipo + r'\b', str(x).lower()))
    )

# Columna con el tipo más frecuente
def tipo_dominante(row, tipos):
    conteos = {t: row[f'count_{t}'] for t in tipos}
    max_val = max(conteos.values())
    if max_val == 0:
        return 'none'
    return max(conteos, key=lambda t: conteos[t])

def segundo_tipo_dominante(row, tipos):
    conteos = {t: row[f'count_{t}'] for t in tipos}
    sorted_tipos = sorted(conteos.items(), key=lambda x: x[1], reverse=True)
    if sorted_tipos[1][1] == 0:
        return 'none'
    return sorted_tipos[1][0]

print("\n[3.2] Calculando tipo dominante y segundo tipo dominante...")
df['tipo_dominante'] = df.apply(lambda row: tipo_dominante(row, TIPOS_POKEMON), axis=1)
df['segundo_tipo'] = df.apply(lambda row: segundo_tipo_dominante(row, TIPOS_POKEMON), axis=1)

print("\n  Distribución de tipos dominantes:")
print(df['tipo_dominante'].value_counts().to_string())

# --- 3.3 Agregar verificación a los clusters P1 y P2 ---
print("\n[3.3] Agregando verificación a los clusters de P1 y P2...")

# DataFrame final P1 con verificación
resultado_p1_verificado = df[['Pokemon', 'cluster_p1', 'tipo_dominante', 'segundo_tipo']].copy()
resultado_p1_verificado.to_csv('resultado_p1_verificado.csv', index=False)

# DataFrame final P2 con verificación
resultado_p2_verificado = df[['Pokemon', 'cluster_p2', 'tipo_dominante', 'segundo_tipo']].copy()
resultado_p2_verificado.to_csv('resultado_p2_verificado.csv', index=False)

print("\n  Archivos guardados: 'resultado_p1_verificado.csv', 'resultado_p2_verificado.csv'")

# --- 3.4 Interpretación de clusters P1 con verificación ---
print("\n[3.4] INTERPRETACIÓN DE CLUSTERS - PREGUNTA 1 (con verificación):")
interpretaciones_p1 = {}
for c in range(N_CLUSTERS_P1):
    subset = resultado_p1_verificado[resultado_p1_verificado['cluster_p1'] == c]
    tipos_freq = subset['tipo_dominante'].value_counts()
    top_tipo = tipos_freq.index[0] if len(tipos_freq) > 0 else 'N/A'
    pct = (tipos_freq.iloc[0] / len(subset) * 100) if len(subset) > 0 else 0
    interpretaciones_p1[c] = top_tipo
    print(f"  Cluster {c:2d} ({len(subset):3d} Pokémon) → Tipo dominante: {top_tipo:10s} ({pct:.1f}%)")
    print(f"           Distribución: {dict(tipos_freq.head(3))}")

# --- 3.5 Interpretación de clusters P2 con verificación ---
print("\n[3.5] INTERPRETACIÓN DE CLUSTERS - PREGUNTA 2 (con verificación):")
interpretaciones_p2 = {}
for c in range(N_CLUSTERS_P2):
    subset = resultado_p2_verificado[resultado_p2_verificado['cluster_p2'] == c]
    tipos_freq = subset['tipo_dominante'].value_counts()
    top_tipo = tipos_freq.index[0] if len(tipos_freq) > 0 else 'N/A'
    pct = (tipos_freq.iloc[0] / len(subset) * 100) if len(subset) > 0 else 0
    interpretaciones_p2[c] = top_tipo
    print(f"  Cluster {c:2d} ({len(subset):3d} Pokémon) → Tipo dominante: {top_tipo:10s} ({pct:.1f}%)")
    print(f"           Distribución: {dict(tipos_freq.head(3))}")

# --- 3.6 Resumen final con nombres de clusters ---
print("\n[3.6] RESUMEN FINAL - NOMBRE DE CADA CLUSTER:")
print("\n  Pregunta 1 (TF-IDF general con bigramas):")
for c, tipo in interpretaciones_p1.items():
    nombre = f"Cluster {c}: '{tipo.upper()}'"
    print(f"  {nombre}")

print("\n  Pregunta 2 (Vocabulario controlado - 18 tipos):")
for c, tipo in interpretaciones_p2.items():
    nombre = f"Cluster {c}: '{tipo.upper()}'"
    print(f"  {nombre}")

# --- 3.7 DataFrame completo final ---
df_final = df[['Pokemon', 'cluster_p1', 'cluster_p2', 'tipo_dominante', 'segundo_tipo']].copy()
df_final['nombre_cluster_p1'] = df_final['cluster_p1'].map(interpretaciones_p1)
df_final['nombre_cluster_p2'] = df_final['cluster_p2'].map(interpretaciones_p2)
df_final.to_csv('resultado_final_completo.csv', index=False)
print("\n[3.7] Archivo 'resultado_final_completo.csv' guardado con toda la información.")

print("\n" + "=" * 70)
print("EJECUCIÓN COMPLETADA EXITOSAMENTE")
print("=" * 70)
print("\nArchivos generados:")
print("  - resultado_pregunta1.csv     → Pokémon + cluster P1")
print("  - resultado_pregunta2.csv     → Pokémon + cluster P2")
print("  - resultado_p1_verificado.csv → P1 + tipo dominante")
print("  - resultado_p2_verificado.csv → P2 + tipo dominante")
print("  - resultado_final_completo.csv → Resumen completo")
