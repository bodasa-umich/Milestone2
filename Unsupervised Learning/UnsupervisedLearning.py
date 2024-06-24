import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation, NMF

# Load and clean dataset
df = pd.read_csv('18776_df_group.csv')  # Adjust the path if necessary
df = df[~df['GroupName'].str.startswith('Violations')]

# Display the first few rows of the dataframe
print(df.head())

# Text cleaning function
def clean_text(text):
    return text.lower()

df['cleaned_text'] = df['grouped_text'].apply(clean_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(df['cleaned_text'])

# Normalize the data
X_norm = normalize(X)

# Apply TruncatedSVD
svd = TruncatedSVD(n_components=50, random_state=42)
X_svd = svd.fit_transform(X_norm)

# Plot explained variance
explained_variance = svd.explained_variance_ratio_.cumsum()
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.axhline(y=0.9, color='r', linestyle='--')
plt.title('Explained Variance by TruncatedSVD Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

# K-Means clustering
num_clusters = 7
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_svd)
silhouette_avg = silhouette_score(X_svd, df['Cluster'])
print(f"Silhouette Score for K-Means: {silhouette_avg}")

# LDA topic modeling
lda = LatentDirichletAllocation(n_components=num_clusters, random_state=42)
lda_topics = lda.fit_transform(X)

# NMF topic modeling
nmf = NMF(n_components=num_clusters, random_state=42)
nmf_topics = nmf.fit_transform(X)

# Visualization: t-SNE for K-Means
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_svd)
plt.figure(figsize=(12, 8))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=df['Cluster'], palette='tab10')
plt.title('t-SNE visualization of K-Means Clusters')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Visualization: Word Clouds for LDA and NMF topics
def plot_word_clouds(model, num_topics, title):
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(model.components_):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join([feature_names[i] for i in topic.argsort()[:-11:-1]]))
        plt.figure(figsize=(2, 1))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {title} Topic {topic_idx}')
        plt.show()

plot_word_clouds(lda, num_clusters, "LDA")
plot_word_clouds(nmf, num_clusters, "NMF")

# Sensitivity Analysis: Varying number of clusters for K-Means
silhouette_scores = []
for k in range(2,8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_svd)
    silhouette_avg = silhouette_score(X_svd, clusters)
    silhouette_scores.append(silhouette_avg)
    print(f"Silhouette Score for K-Means with {k} clusters: {silhouette_avg}")

plt.figure(figsize=(10, 6))
plt.plot(range(2, 8), silhouette_scores, marker='o')
plt.title('Silhouette Scores for Varying Number of Clusters (K-Means)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

# Top words per topic
def plot_top_words(model, feature_names, num_top_words, title):
    fig, axes = plt.subplots(1, num_clusters, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-num_top_words - 1::-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx + 1}', fontdict={'fontsize': 15})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=12)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
    fig.suptitle(title, fontsize=20)
    plt.subplots_adjust(top=0.9, bottom=0.05, wspace=0.5, hspace=0.3)
    plt.show()

plot_top_words(lda, vectorizer.get_feature_names_out(), 10, 'Top words per LDA topic')
plot_top_words(nmf, vectorizer.get_feature_names_out(), 10, 'Top words per NMF topic')

# Overall Summary
summary_table = pd.DataFrame({
    "Method": ["K-Means", "LDA", "NMF"],
    "Silhouette Score": [silhouette_score(X_svd, df['Cluster']), "N/A", "N/A"],
    "Topic Coherence": ["N/A", "Calculated Coherence Score", "Calculated Coherence Score"]
})
print(summary_table)

# Sensitivity Analysis: Varying number of topics for LDA and NMF
topics = [3, 4, 5, 6, 7]
lda_perplexity = []
nmf_reconstruction_error = []

for n_topics in topics:
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    lda_perplexity.append(lda.perplexity(X))
    
    nmf = NMF(n_components=n_topics, random_state=42)
    W = nmf.fit_transform(X)
    H = nmf.components_
    X_dense = X.toarray() if hasattr(X, 'toarray') else X
    nmf_reconstruction_error.append(np.sqrt(np.sum((X_dense - np.dot(W, H)) ** 2)))

plt.figure(figsize=(10, 6))
plt.plot(topics, lda_perplexity, marker='o', label='LDA Perplexity')
plt.plot(topics, nmf_reconstruction_error, marker='o', label='NMF Reconstruction Error')
plt.title('Scores for Varying Number of Topics (LDA and NMF)')
plt.xlabel('Number of Topics')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()
