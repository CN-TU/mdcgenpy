import json
from ..clusters import ClusterGenerator


def get_cluster_generator(input_file):
    """
    Parses a JSON file and generates a ClusterGenerator.
    Args:
        input_file (str or file): Input file.

    Returns:
        ClusterGenerator: Resulting ClusterGenerator from the input file.
    """
    try:
        data = input_file.read()
    except AttributeError:
        data = open(input_file).read()
    data = json.loads(data)

    out = ClusterGenerator(**data)

    if 'clusters' not in data:
        return out

    if len(data['clusters']) > out.n_clusters:
        raise IOError('Invalid input! Number of clusters is smaller than clusters defined in input!')

    for clust_data, clust in zip(data['clusters'], out.clusters):
        for key in clust_data:
            clust[key] = clust_data[key]

    return out
