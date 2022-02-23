import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import cosine


def jsd(p1, p2) -> float:
    '''Returns the Jensen Shannon Divergence'''
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p1 /= p1.sum()
    p2 /= p2.sum()
    m = (p1 + p2) / 2
    return (entropy(p1, m) + entropy(p2, m)) / 2


def cosine_similarity(a, b) -> float:
    '''Returns the Cosine Similarity'''
    a = np.asarray(a)
    b = np.asarray(b)
    return 1 - cosine(a, b)


def cosine_distance(a, b) -> float:
    '''Returns the Cosine Distance'''
    a = np.asarray(a)
    b = np.asarray(b)
    return cosine(a, b)


def div(a, b) -> float:
    '''Returns the Difference Between Diversities (DIV)'''
    a = np.asarray(a)
    b = np.asarray(b)

    # centroids
    muA, muB = a.mean(axis=0), b.mean(axis=0)

    # diversities
    divA = np.array([cosine_distance(x, muA) for x in a])
    divB = np.array([cosine_distance(x, muB) for x in b])

    return abs(divA.mean(axis=0) - divB.mean(axis=0))


def pdis(a, b, label_a, label_b) -> float:
    '''Return the Cosine Distance between prototype embeddings'''

    # unique labels and size
    unique_a, counts_a = np.unique(label_a, return_counts=True)
    unique_b, counts_b = np.unique(label_b, return_counts=True)

    # cluster centroids
    mu_a = np.array([a[label_a == label, :].mean(axis=0) for label in unique_a])
    mu_b = np.array([b[label_b == label, :].mean(axis=0) for label in unique_b])

    return cosine_distance(mu_a.mean(axis=0), mu_b.mean(axis=0))


def pdiv(a, b, label_a, label_b) -> float:
    '''Difference between prototype embedding diversities'''

    # unique labels and size
    unique_a, counts_a = np.unique(label_a, return_counts=True)
    unique_b, counts_b = np.unique(label_b, return_counts=True)

    # cluster centroids
    mu_a = np.array([a[label_a == label, :].mean(axis=0) for label in unique_a])
    mu_b = np.array([b[label_b == label, :].mean(axis=0) for label in unique_b])

    return div(mu_a, mu_b)
