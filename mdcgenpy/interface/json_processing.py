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
    data = get_input_data(input_file)

    out = ClusterGenerator(**data)

    if 'clusters' not in data:
        return out

    if len(data['clusters']) > out.n_clusters:
        raise IOError('Invalid input! Number of clusters is smaller than clusters defined in input!')

    for clust_data, clust in zip(data['clusters'], out.clusters):
        for key in clust_data:
            setattr(clust, key, clust_data[key])

    return out


def get_input_data(input_file):
    """
    Helper function for get_cluster_generator()
    Args:
        input_file (str or file): Input file.

    Returns:
        dict: data corresponding to JSON input.
    """
    try:  # if input_file is string in json format
        data = json.loads(input_file.decode('string_escape'))
    except TypeError:
        data = json.loads(input_file.decode())
    except:
        try:  # if input_file is an open file
            data = input_file.read()
        except AttributeError:  # if input_file is a filename
            data = open(input_file).read()
        data = json.loads(data)
    return data
