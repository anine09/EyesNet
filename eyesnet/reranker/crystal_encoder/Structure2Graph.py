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

def get_gotennet_neighbors_info(structure, cutoff=5.0):
    """
    Get the gotennet neighbors of a structure.
    """
    neighbors = structure.get_all_neighbors(cutoff)
    edge_index_list = []
    r_ij_list = []
    dir_ij_list = []

# 遍历每个原子及其邻居
    for i, atom_neighbors in enumerate(neighbors):
        for neighbor in atom_neighbors:
            j = neighbor.index  # 邻居的索引
            distance = neighbor.nn_distance  # 邻居的距离
            displacement = neighbor.coords - structure[i].coords  # 邻居的位移向量

            # 添加到列表
            edge_index_list.append([int(i), int(j)])
            r_ij_list.append(float(distance))
            dir_ij_list.append((displacement / distance).tolist())  # 单位向量
    return edge_index_list, r_ij_list, dir_ij_list

def get_gotennet_atomic_numbers(structure):
    """
    Get the atomic numbers of a structure.
    """
    atomic_numbers = [site.specie.Z for site in structure]
    return atomic_numbers

def get_gotennet_pos(structure):
    """
    Get the positions of a structure.
    """
    pos = structure.cart_coords.tolist()
    return pos