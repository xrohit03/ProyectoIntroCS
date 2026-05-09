import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# ====================================================================
# CARGA DEL DATASET
# ====================================================================
data_frame = pd.read_csv('smogon.csv')
data_frame.columns = data_frame.columns.str.strip().str.lower()

# Remplazamos del Data frame los valores vacíos(NaN) por texto vacío.
data_frame['moves'] = data_frame['moves'].fillna('')


#   
my_stop_words = list(ENGLISH_STOP_WORDS)  # lista de stop words
my_stop_words.remove("fire")              # eliminamos fire de los stop words
my_stop_words.append("pp")                # "pp" no estará dentro del vocabulario
my_stop_words.append("power")             # "power" no estará dentro del vocabulario
my_stop_words.append("accuracy")          # "accuracy" no estará dentro del vocabulario


# =====================================================================
# [1.1]  Generar la matriz tf-idf utilizando una cantidad de bi-gramas
# =====================================================================

# Usamos unigramas + bigramas para capturar frases como "fire punch", "water gun", etc.
tf_idf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),             # se acepta unigramas y bigramas
    stop_words=my_stop_words,       # palabras a no tomar en cuenta
    max_features=5000,              # limitamos vocabulario
    min_df=2,                       # el n-grama debe aparecer en al menos 2 pokemones
    sublinear_tf=True,              # educir la diferencia entre las frecuencias con log(tf)
    token_pattern=r'[a-zA-Z]{3,}'   # solo tokens alfabéticos de al menos 3 letras
)

# Definimos variables
ataques = data_frame['moves'].tolist()
matriz_tf_idf = tf_idf_vectorizer.fit_transform(ataques)
vocabulario = tf_idf_vectorizer.get_feature_names_out()

# ============================================================
# [1.2] Mostrar el número total de columnas
# ============================================================
print(f"\nNúmero total de columnas (elementos del vocabulario): {len(vocabulario)}")

# ============================================================
# [1.3] Imprimir los elementos del vocabulario
# ============================================================
print(f"\nLos primeros 50 elementos del vocabulario:")
for i, elemento in enumerate(vocabulario[:50]):
    print(f"{i+1:2d}. {elemento}")

print(f"  ... (total: {len(vocabulario)} elementos)\n")

# ============================================================
# [1.4] Generar un DataFrame con la matriz tf-idf que tenga como 
# cabeceras los elementos de su vocabulario. Imprimir dicha matriz
# ============================================================
data_frame_matriz_tf_idf = pd.DataFrame(
    matriz_tf_idf.toarray(), 
    data_frame['pokemon'], 
    vocabulario,
)
#Imprimimos el Dataframe realizado con la matriz TF-IDF
print(data_frame_matriz_tf_idf)



# ============================================================
# [1.5] Clustering con K-Means
# ============================================================

# Normalizamos para mejor clustering con TF-IDF
tf_idf_normalizado = normalize(matriz_tf_idf)

# Creamos el objeto K_Means
k_means = KMeans(
    n_clusters=18,
    random_state=42,
    n_init=20,
    max_iter=500
)
# Aplica K-Means a la matriz TF-IDF normalizada
k_means.fit(tf_idf_normalizado)


# Obtiene el cluster de cada Pokémon y lo guardamos 
# en el Dataframe
labels = k_means.labels_
data_frame['cluster'] = labels

# ============================================================
# [1.6] Guardamos el CSV con Pokémon y cluster
# ============================================================
resultado = data_frame[['pokemon', 'cluster']].copy()
resultado.to_csv('resultado_pregunta1.csv', index=False)


# ============================================================
# [1.7] Mostramos el contenido de cada cluster
# ============================================================
for cluster in range(18):
    pokemon_cluster = data_frame[data_frame['cluster'] == cluster]['pokemon'].tolist()
    print(f"\nCluster {cluster} ({len(pokemon_cluster)} Pokémon):")
    print(f"{', '.join(pokemon_cluster[:20])}", end='')
    if len(pokemon_cluster) > 20:
        print(f"... (+{len(pokemon_cluster)-20} más)")

# Mostramos 10 terminos más relevantes de cada cluster
print("\nTop 10 términos TF-IDF más relevantes por cluster (P1):")
order_centroids = k_means.cluster_centers_.argsort()[:, ::-1]
terms = tf_idf_vectorizer.get_feature_names_out()

for cluster in range(18):
    top_terms = [terms[ind] for ind in order_centroids[cluster, :10]]
    print(f"Cluster {cluster}: {', '.join(top_terms)}")