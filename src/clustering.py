from sklearn.cluster import KMeans

def cluster_features(features, k=3):
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(features)
    return labels