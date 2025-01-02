from pymatgen.analysis.local_env import CrystalNN

cnn = CrystalNN()


def get_adjacency_matrix(structure):
    """
    Get the adjacency matrix of a structure.
    """
    adjacency_matrix = [[False] * len(structure) for _ in range(len(structure))]

    for i in range(len(structure)):
        neighbors = cnn.get_nn_info(structure, i)
        for neighbor in neighbors:
            j = neighbor["site_index"]
            adjacency_matrix[i][j] = True
            adjacency_matrix[j][i] = True

    return adjacency_matrix


def get_atom_ids(structure):
    """
    Get the atom ids of a structure.
    """
    atom_ids = [site.specie.number for site in structure]
    return atom_ids


def get_coordinates(structure):
    """
    Get the coordinates of a structure.
    """
    coordinates = [site.coords.tolist() for site in structure]
    return coordinates
