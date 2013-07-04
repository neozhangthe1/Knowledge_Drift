def jaccard_similarity(d1, d2):
    s1 = set(d1)
    s2 = set(d2)
    intersect = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return float(intersect) / union



