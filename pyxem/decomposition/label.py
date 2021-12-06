import numpy as np
from scipy.spatial import distance_matrix

def spatial_cluster(vectors, max_distances=1, axes=(0,1),):
    """ Cluster based on distance between vectors.  Only the axes in
    Axes are considered.

    Pameters
    --------
    vectors: Array-like
        The list of all of the vectors to be considered.  Each vector is compared to


    """
    return


def refine_labels(vectors,
                  labels,
                  navigation_indexes=(0, 1),
                  max_distances=1,
                  axes=(0, 1),
                  symmetries=(2, 4, 6, 10),
                  structural_sym=None,
                  ):
    """ Cluster based on distance between vectors.  Only the axes in
    Axes are considered.

    Pameters
    --------
    vectors: Array-like
        An array of labeled vectors to be compared and potentially refined/ reduced where
        some label refers to the same feature
    labels: Array-like
        An array of the labels for the different vectors.  Used as a starting point for refining labels
    max_distance: float, tuple
        The max_distance along each of the axes of comparison to be considered the "Same" feature
    symmetries: None, tuple
        The list of symmetry operations to preform on the data. Labels some symmetry operation
        and within max_distance for the radial axis will be grouped
    structural_sym: None, float
        If not None then the extent of two labels will be compared.
        Any labels with a structural similarity larger than
        structural_sym will be combined.
    """
    centers = [get_label_center(vectors[labels == l], axes=axes) for l in np.unique(labels)]
    distance = distance_matrix(centers, centers)

    groups = np.where()
    num_inside = np.sum([distance < max_distance], axis=1)[0]
    nearest = np.transpose(np.argsort(distance, axis=0))
    nearest = [n[1:int(num)] for n, num in zip(nearest, num_inside)]
    grouped_clusters = [[clusters[int(n)] for n in near] for near in nearest]
    angles = [get_angles(c) for c in grouped_clusters]
    char_angles = get_characteristic_angles()
    char_angles = [[a, 0] for a in char_angles]
    include_cluster = [is_close(a, char_angles) if a is not None else [True] for a in angles]
    gc = [[cl for cl, g in zip(c, gr) if g] for c, gr in zip(grouped_clusters, include_cluster)]
    return

def get_label_center(vectors, axes):
    return np.mean(vectors[axes])


def get_characteristic_angles(syms=(2,4,6,10)):
    fract = [i/sy * np.pi for sy in syms for i in range(sy) ]
    return list(set(fract))

def get_angles(clusters):
    if len (clusters) ==0:
        return None
    pixels = [[c.real_indexes[3],0] for c in clusters]
    dist = distance_matrix([pixels[0],], pixels)[0]
    return [[d,0]for d in dist]