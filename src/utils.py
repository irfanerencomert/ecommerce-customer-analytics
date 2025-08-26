import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans


def evaluate_clusters(data, max_clusters=10):
    """Optimum küme sayısını belirler"""
    scores = []
    K_range = range(2, max_clusters + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)

        if k > 1:  # Silhouette skoru için en az 2 küme gerekir
            sil_score = silhouette_score(data, labels)
            ch_score = calinski_harabasz_score(data, labels)
            scores.append({'K': k, 'Silhouette': sil_score, 'Calinski': ch_score})

    scores_df = pd.DataFrame(scores)

    # Görselleştirme
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(scores_df['K'], scores_df['Silhouette'], 'b-', label='Silhouette')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Silhouette Score', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(scores_df['K'], scores_df['Calinski'], 'r-', label='Calinski-Harabasz')
    ax2.set_ylabel('Calinski-Harabasz Score', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.title('Cluster Evaluation Metrics')
    plt.savefig('../data/processed/cluster_evaluation.png')

    return scores_df