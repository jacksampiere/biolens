from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def encoder(X, k):
    """Spoof encoder E by returning first k elements of each input as embeddings.

    Args:
        X: Array of inputs.
        k: Embedding dimension to retain.
    Returns:
        Array of embeddings with k elements.
    """
    return X[:, :k]


def fit_preproc_map(emb):
    """Fit preprocessing map ϕ (z-score) on embeddings before projection.

    Args:
        emb: Embeddings array.
    Returns:
        Fitted StandardScaler object.
    """
    preproc_map = StandardScaler()
    preproc_map.fit(X=emb)
    return preproc_map


def apply_preproc_map(preproc_map, emb):
    """Apply preprocessing map ϕ to embeddings.

    Args:
        preproc_map: Fitted preprocessing map object.
        emb: Embeddings array.
    Returns:
        Preprocessed embeddings.
    """
    return preproc_map.transform(X=emb)


def fit_projection(emb, m):
    """Fit projection P to reduce embeddings to m dims (MVP uses PCA1).

    Args:
        emb: Embeddings array (typically preprocessed).
        m: Target projection dimension.
    Returns:
        Fitted PCA object.
    """
    projection = PCA(n_components=m, random_state=42)
    projection.fit(X=emb)
    return projection


def apply_projection(projection, emb):
    """Apply projection P to embeddings; MVP scalarizes to 1D.

    Args:
        projection: Fitted projection object.
        emb: Embeddings array.
    Returns:
        Projected embeddings.
    """
    return projection.transform(X=emb).ravel()
