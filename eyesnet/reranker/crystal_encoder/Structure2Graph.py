import torch
import numpy as np
from pymatgen.analysis.local_env import CrystalNN

cnn = CrystalNN()


def get_ajacency_matrix(structure):
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
    adjacency_matrix = torch.tensor(adjacency_matrix)
    return adjacency_matrix


def get_atom_ids(structure):
    """
    Get the atom ids of a structure.
    """
    atom_ids = [site.specie.number for site in structure]
    atom_ids = torch.tensor(atom_ids)
    return atom_ids


def get_coordinates(structure):
    """
    Get the coordinates of a structure.
    """
    coordinates = np.array([site.coords for site in structure])
    coordinates = torch.tensor(coordinates)
    return coordinates
