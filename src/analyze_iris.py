import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.stats import mode
import warnings
warnings.filterwarnings("ignore")

# Carregando os dados do dataset Iris usando a ucimlrepo
print("Baixando dataset Iris...")
iris = fetch_ucirepo(id=53)
X = iris.data.features.values
y_str = iris.data.targets.values.ravel()

# Como os modelos matemáticos lidam melhor com números, converti as classes textuais para inteiros (0, 1, 2)
classes = np.unique(y_str)
class_map = {c: i for i, c in enumerate(classes)}
y = np.array([class_map[c] for c in y_str])

# Criei esta função auxiliar para mapear os clusters do K-Means para as classes reais (usando a moda)
def map_clusters(y_true, y_pred):
    mapped_labels = np.zeros_like(y_pred)
    for i in range(3):
        mask = (y_pred == i)
        if np.sum(mask) > 0:
            # Descubro a classe que mais apareceu neste cluster e atribuo a ele
            m = mode(y_true[mask], keepdims=True)[0][0]
            mapped_labels[mask] = m
    return mapped_labels

# Primeiro teste: Avaliando meus classificadores no dataset limpo
# SVM (Supervisionado)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
svm_clean = SVC(random_state=42)
svm_clean.fit(X_train, y_train)
y_pred_svm_clean = svm_clean.predict(X_test)
res_svm_clean = {
    'Cenário': 'Limpo', 'Classificador': 'SVM',
    'Recall': round(recall_score(y_test, y_pred_svm_clean, average='macro'), 4),
    'Precision': round(precision_score(y_test, y_pred_svm_clean, average='macro', zero_division=0), 4),
    'F1-Score': round(f1_score(y_test, y_pred_svm_clean, average='macro'), 4)
}

# Para o modelo não supervisionado (K-Means), eu ajusto no conjunto inteiro e depois mapeio as respostas
kmeans_clean = KMeans(n_clusters=3, random_state=42, n_init=10)
y_pred_km_clean = kmeans_clean.fit_predict(X)
y_pred_km_clean_mapped = map_clusters(y, y_pred_km_clean)
res_km_clean = {
    'Cenário': 'Limpo', 'Classificador': 'K-Means',
    'Recall': round(recall_score(y, y_pred_km_clean_mapped, average='macro'), 4),
    'Precision': round(precision_score(y, y_pred_km_clean_mapped, average='macro', zero_division=0), 4),
    'F1-Score': round(f1_score(y, y_pred_km_clean_mapped, average='macro'), 4)
}

# Aqui eu gero 20% de outliers selecionando índices aleatoriamente
np.random.seed(42)
n_samples = X.shape[0]
n_outliers = int(0.20 * n_samples)
outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)

X_dirty = X.copy()
# Escolhi multiplicar os dados sorteados por 5 para jogá-los violentamente para fora da distribuição
X_dirty[outlier_indices] = X_dirty[outlier_indices] * 5.0

# Segundo teste: Avaliando a resiliência dos modelos nos dados corrompidos
# SVM
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_dirty, y, test_size=0.3, random_state=42)
svm_dirty = SVC(random_state=42)
svm_dirty.fit(X_train_d, y_train_d)
y_pred_svm_dirty = svm_dirty.predict(X_test_d)
res_svm_dirty = {
    'Cenário': 'Sujo', 'Classificador': 'SVM',
    'Recall': round(recall_score(y_test_d, y_pred_svm_dirty, average='macro'), 4),
    'Precision': round(precision_score(y_test_d, y_pred_svm_dirty, average='macro', zero_division=0), 4),
    'F1-Score': round(f1_score(y_test_d, y_pred_svm_dirty, average='macro'), 4)
}

# K-Means
kmeans_dirty = KMeans(n_clusters=3, random_state=42, n_init=10)
y_pred_km_dirty = kmeans_dirty.fit_predict(X_dirty)
y_pred_km_dirty_mapped = map_clusters(y, y_pred_km_dirty)
res_km_dirty = {
    'Cenário': 'Sujo', 'Classificador': 'K-Means',
    'Recall': round(recall_score(y, y_pred_km_dirty_mapped, average='macro'), 4),
    'Precision': round(precision_score(y, y_pred_km_dirty_mapped, average='macro', zero_division=0), 4),
    'F1-Score': round(f1_score(y, y_pred_km_dirty_mapped, average='macro'), 4)
}

# Gerando a tabela final e imprimindo no terminal
results = pd.DataFrame([res_km_clean, res_km_dirty, res_svm_clean, res_svm_dirty])
print("\n--- RESULTADOS TABELA ---")
print("| Cenário | Classificador | Recall | Precision | F1-Score |")
print("| --- | --- | --- | --- | --- |")
def fmt(v): return str(v).replace('.', ',')
for _, row in results.iterrows():
    print(f"| {row['Cenário']} | {row['Classificador']} | {fmt(row['Recall'])} | {fmt(row['Precision'])} | {fmt(row['F1-Score'])} |")
print("-------------------------\n")

# Gráfico: PCA para visualizar os outliers vs normais
pca = PCA(n_components=2)
X_pca_dirty = pca.fit_transform(X_dirty)

plt.figure(figsize=(8, 6))
normal_indices = np.setdiff1d(np.arange(n_samples), outlier_indices)
plt.scatter(X_pca_dirty[normal_indices, 0], X_pca_dirty[normal_indices, 1], c='blue', label='Dados Normais', alpha=0.6)
plt.scatter(X_pca_dirty[outlier_indices, 0], X_pca_dirty[outlier_indices, 1], c='red', label='Outliers (Sabotados)', marker='x', s=100)
plt.title('Dispersão do Dataset Iris após injeção de 20% de Outliers (PCA)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('images/iris_dispersao_outliers.png')
print("Gráfico 'iris_dispersao_outliers.png' salvo com sucesso na pasta images/.")
