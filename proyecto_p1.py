"""
CS1000 - Introducción a las Ciencias de la Computación
Proyecto 1 (P1) - 2026-1
Profesor: Ian Paul Brossard
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
# CARGA DEL DATASET
# ============================================================
print("=" * 70)
print("CARGANDO DATASET")
print("=" * 70)

df = pd.read_csv('smogon.csv', on_bad_lines='skip')
df['moves'] = df['moves'].fillna('')
print(f"Dataset cargado: {df.shape[0]} Pokémon, {df.shape[1]} columnas")
print(f"Columnas: {df.columns.tolist()}")
print(f"Valores nulos en 'moves': {df['moves'].isna().sum()}")

# ============================================================
# LOS 18 TIPOS  (y los 17 sin normal para verificación)
# ============================================================
TIPOS_POKEMON = [
    'normal', 'fire', 'water', 'electric', 'grass', 'ice',
    'fighting', 'poison', 'ground', 'flying', 'psychic', 'bug',
    'rock', 'ghost', 'dragon', 'dark', 'steel', 'fairy'
]

# Para la verificación excluimos 'normal' porque aparece en TODOS los
# Pokémon (casi todos los moves genéricos son tipo Normal) y no
# aporta información para distinguir grupos.
TIPOS_SIN_NORMAL = [t for t in TIPOS_POKEMON if t != 'normal']

# ============================================================
# FUNCIÓN DE PREPROCESAMIENTO CORREGIDA
# ============================================================
def preprocesar_moves(texto, tipos):
    """
    Separa cada tipo de su contexto pegado usando dos estrategias:
    1. Reemplaza \\xa0 (espacio no rompible) por espacio normal.
    2. Inserta espacio ANTES de aplicar .lower(), buscando el tipo
       con la primera letra en mayúscula (como aparece en el CSV:
       'AttractNormal', 'BiteFirele', 'Body SlamNormal', etc.)
    """
    texto = str(texto)
    texto = texto.replace('\xa0', ' ')          # espacio no rompible → normal
    for tipo in tipos:
        # Busca "Fire", "Water", "Normal"... y les pone espacio
        texto = re.sub(tipo.capitalize(), f' {tipo.capitalize()} ', texto)
    texto = texto.lower()                        # ahora sí a minúsculas
    return texto

# ============================================================
# PREGUNTA 1: TF-IDF CON N-GRAMAS
# ============================================================
print("\n" + "=" * 70)
print("PREGUNTA 1: AGRUPAMIENTO MEDIANTE TF-IDF CON N-GRAMAS")
print("=" * 70)

print("\n[1.1] Generando matriz TF-IDF con unigramas y bigramas...")

tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    min_df=2,
    sublinear_tf=True,
    token_pattern=r'[a-zA-Z]{2,}'
)

moves_corpus = df['moves'].tolist()
tfidf_matrix = tfidf_vectorizer.fit_transform(moves_corpus)
vocabulario = tfidf_vectorizer.get_feature_names_out()

print(f"\n[1.2] Número total de columnas (elementos del vocabulario): {len(vocabulario)}")

print(f"\n[1.3] Primeros 100 elementos del vocabulario:")
for i, term in enumerate(vocabulario[:100]):
    print(f"  {i+1:3d}. {term}")
print(f"  ... (total: {len(vocabulario)} elementos)")

print("\n[1.4] Generando DataFrame con la matriz TF-IDF...")
df_tfidf = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vocabulario,
    index=df['Pokemon']
)
print("Imprimiendo matriz TF-IDF (print):")
print(df_tfidf)
print(f"\nDimensiones del DataFrame TF-IDF: {df_tfidf.shape}")

print(f"\n[1.5] Aplicando K-Means con 18 clusters...")
tfidf_norm = normalize(tfidf_matrix)
kmeans_p1 = KMeans(n_clusters=18, random_state=42, n_init=20, max_iter=500)
kmeans_p1.fit(tfidf_norm)
df['cluster_p1'] = kmeans_p1.labels_

resultado_p1 = df[['Pokemon', 'cluster_p1']].copy()
resultado_p1.to_csv('resultado_pregunta1.csv', index=False)
print("\n[1.6] Archivo 'resultado_pregunta1.csv' guardado.")

print("\n[1.7] Contenido de cada cluster (Pregunta 1):")
for c in range(18):
    poks = df[df['cluster_p1'] == c]['Pokemon'].tolist()
    print(f"\n  Cluster {c} ({len(poks)} Pokémon):")
    print(f"  {', '.join(poks[:20])}", end='')
    if len(poks) > 20:
        print(f" ... (+{len(poks)-20} más)")
    else:
        print()

print("\n[1.7b] Top 10 términos TF-IDF más relevantes por cluster (P1):")
order_centroids = kmeans_p1.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vectorizer.get_feature_names_out()
for c in range(18):
    top_terms = [terms[ind] for ind in order_centroids[c, :10]]
    print(f"  Cluster {c}: {', '.join(top_terms)}")

# ============================================================
# PREGUNTA 2: VOCABULARIO CONTROLADO
# ============================================================
print("\n" + "=" * 70)
print("PREGUNTA 2: AGRUPAMIENTO CON VOCABULARIO CONTROLADO (18 TIPOS)")
print("=" * 70)

print("\n[2.1] Preprocesando texto con función corregida...")
df['moves_procesado'] = df['moves'].apply(
    lambda x: preprocesar_moves(x, TIPOS_POKEMON)
)

# Ejemplo
print("\n  Ejemplo (Arcanine, primeros 400 chars):")
print(f"  ORIGINAL:  {df[df['Pokemon']=='Arcanine']['moves'].iloc[0][:300]}")
print(f"  PROCESADO: {df[df['Pokemon']=='Arcanine']['moves_procesado'].iloc[0][:300]}")

print("\n[2.2] Generando TF-IDF con vocabulario controlado (18 tipos)...")
tfidf_vocab = TfidfVectorizer(
    vocabulary=TIPOS_POKEMON,
    token_pattern=r'[a-zA-Z]+'
)
tfidf_matrix_p2 = tfidf_vocab.fit_transform(df['moves_procesado'].tolist())

df_tfidf_p2 = pd.DataFrame(
    tfidf_matrix_p2.toarray(),
    columns=TIPOS_POKEMON,
    index=df['Pokemon']
)
print(f"\n  Dimensiones matriz TF-IDF vocabulario controlado: {df_tfidf_p2.shape}")
print("\n  Primeras 10 filas:")
print(df_tfidf_p2.head(10))

print(f"\n[2.4] Aplicando K-Means con 18 clusters...")
tfidf_norm_p2 = normalize(tfidf_matrix_p2)
kmeans_p2 = KMeans(n_clusters=18, random_state=42, n_init=50, max_iter=500)
kmeans_p2.fit(tfidf_norm_p2)
df['cluster_p2'] = kmeans_p2.labels_

resultado_p2 = df[['Pokemon', 'cluster_p2']].copy()
resultado_p2.to_csv('resultado_pregunta2.csv', index=False)
print("\n[2.5] Archivo 'resultado_pregunta2.csv' guardado.")

print("\n[2.6] Clusters (Pregunta 2) con tipos dominantes:")
for c in range(18):
    idx = df[df['cluster_p2'] == c].index
    poks = df[df['cluster_p2'] == c]['Pokemon'].tolist()
    puntajes = df_tfidf_p2.iloc[idx].mean()
    top_tipos = puntajes[puntajes > 0].sort_values(ascending=False).head(3)
    tipos_str = ', '.join([f"{t}={v:.3f}" for t, v in top_tipos.items()])
    print(f"\n  Cluster {c} ({len(poks)} Pokémon) → Tipos: {tipos_str}")
    print(f"  Pokémon: {', '.join(poks[:15])}", end='')
    if len(poks) > 15:
        print(f" ... (+{len(poks)-15} más)")
    else:
        print()

# ============================================================
# PREGUNTA 3: VERIFICACIÓN
# ============================================================
print("\n" + "=" * 70)
print("PREGUNTA 3: VERIFICACIÓN DE AGRUPAMIENTO POR TIPOS")
print("=" * 70)

print("\n[3.1] Contando ocurrencias de cada tipo (excluyendo 'normal')...")
for tipo in TIPOS_POKEMON:
    df[f'count_{tipo}'] = df['moves_procesado'].apply(
        lambda x: len(re.findall(r'\b' + tipo + r'\b', str(x)))
    )

def tipo_dominante(row, tipos):
    conteos = {t: row[f'count_{t}'] for t in tipos}
    max_val = max(conteos.values())
    if max_val == 0:
        return 'none'
    return max(conteos, key=lambda t: conteos[t])

def segundo_tipo_dominante(row, tipos):
    conteos = {t: row[f'count_{t}'] for t in tipos}
    sorted_tipos = sorted(conteos.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_tipos) < 2 or sorted_tipos[1][1] == 0:
        return 'none'
    return sorted_tipos[1][0]

# Usamos TIPOS_SIN_NORMAL para que normal no tape a todos los demás
print("\n[3.2] Calculando tipo dominante (sin contar 'normal')...")
df['tipo_dominante'] = df.apply(lambda row: tipo_dominante(row, TIPOS_SIN_NORMAL), axis=1)
df['segundo_tipo']   = df.apply(lambda row: segundo_tipo_dominante(row, TIPOS_SIN_NORMAL), axis=1)

print("\n  Distribución de tipos dominantes (sin normal):")
print(df['tipo_dominante'].value_counts().to_string())

# Verificación P1
print("\n[3.4] INTERPRETACIÓN DE CLUSTERS - PREGUNTA 1 (con verificación):")
interpretaciones_p1 = {}
for c in range(18):
    subset = df[df['cluster_p1'] == c]
    tipos_freq = subset['tipo_dominante'].value_counts()
    top_tipo = tipos_freq.index[0] if len(tipos_freq) > 0 else 'N/A'
    pct = tipos_freq.iloc[0] / len(subset) * 100
    interpretaciones_p1[c] = top_tipo
    print(f"  Cluster {c:2d} ({len(subset):3d} Pokémon) → Tipo dominante: {top_tipo:10s} ({pct:.1f}%)")
    print(f"           Distribución: {dict(tipos_freq.head(3))}")

# Verificación P2
print("\n[3.5] INTERPRETACIÓN DE CLUSTERS - PREGUNTA 2 (con verificación):")
interpretaciones_p2 = {}
for c in range(18):
    subset = df[df['cluster_p2'] == c]
    tipos_freq = subset['tipo_dominante'].value_counts()
    top_tipo = tipos_freq.index[0] if len(tipos_freq) > 0 else 'N/A'
    pct = tipos_freq.iloc[0] / len(subset) * 100
    interpretaciones_p2[c] = top_tipo
    print(f"  Cluster {c:2d} ({len(subset):3d} Pokémon) → Tipo dominante: {top_tipo:10s} ({pct:.1f}%)")
    print(f"           Distribución: {dict(tipos_freq.head(3))}")

# Archivos finales
resultado_p1_v = df[['Pokemon', 'cluster_p1', 'tipo_dominante', 'segundo_tipo']].copy()
resultado_p2_v = df[['Pokemon', 'cluster_p2', 'tipo_dominante', 'segundo_tipo']].copy()
resultado_p1_v.to_csv('resultado_p1_verificado.csv', index=False)
resultado_p2_v.to_csv('resultado_p2_verificado.csv', index=False)

df_final = df[['Pokemon', 'cluster_p1', 'cluster_p2', 'tipo_dominante', 'segundo_tipo']].copy()
df_final['nombre_cluster_p1'] = df_final['cluster_p1'].map(interpretaciones_p1)
df_final['nombre_cluster_p2'] = df_final['cluster_p2'].map(interpretaciones_p2)
df_final.to_csv('resultado_final_completo.csv', index=False)

print("\n[3.6] RESUMEN FINAL - NOMBRE DE CADA CLUSTER:")
print("\n  Pregunta 1 (TF-IDF general con bigramas):")
for c, tipo in interpretaciones_p1.items():
    print(f"  Cluster {c}: '{tipo.upper()}'")

print("\n  Pregunta 2 (Vocabulario controlado - 18 tipos):")
for c, tipo in interpretaciones_p2.items():
    print(f"  Cluster {c}: '{tipo.upper()}'")

print("\n" + "=" * 70)
print("EJECUCIÓN COMPLETADA")
print("=" * 70)
print("\nArchivos generados:")
print("  - resultado_pregunta1.csv")
print("  - resultado_pregunta2.csv")
print("  - resultado_p1_verificado.csv")
print("  - resultado_p2_verificado.csv")
print("  - resultado_final_completo.csv")
