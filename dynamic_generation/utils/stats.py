from statistics import NormalDist


# Copied from https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
def confidence_interval(data, confidence=0.95):
    dist = NormalDist.from_samples(data)
    z = NormalDist().inv_cdf((1 + confidence) / 2.0)
    h = dist.stdev * z / ((len(data) - 1) ** 0.5)
    return dist.mean - h, dist.mean + h
