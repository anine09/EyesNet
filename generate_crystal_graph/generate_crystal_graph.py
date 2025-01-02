import json
import os
import sys
sys.path.insert(0, "/home/lxt/EyesNet/")
from eyesnet.reranker.crystal_encoder.Structure2Graph import (
    get_adjacency_matrix,
    get_atom_ids,
    get_coordinates,
)
from pymatgen.core import Structure
from tqdm import tqdm

def generate_crystal_graph(file_path):
    data = json.load(open(file_path, "r"))
    crystal_structure = Structure.from_str(data["structure"], fmt="json")
    atom_ids = get_atom_ids(crystal_structure)
    coordinates = get_coordinates(crystal_structure)
    adjacency_matrix = get_adjacency_matrix(crystal_structure)
    crystal_data = {
        "atom_ids": atom_ids,
        "coordinates": coordinates,
        "adjacency_matrix": adjacency_matrix,
    }
    data["crystal_graph"] = crystal_data
    return data

if __name__ == "__main__":
    DATASET_PATH = "/home/lxt/EyesNet/dataset/xrd_simulation_data/XRD_exp/"
    TARGET_PATH = "/home/lxt/EyesNet/dataset/crystal_graph/XRD_exp/"

    already_processed = []

    for *_, files in os.walk(TARGET_PATH):
        for file in files:
            already_processed.append(file)

    for *_, files in os.walk(DATASET_PATH):
        for file in tqdm(files):
            if file in already_processed:
                continue
            try:
                data = generate_crystal_graph(DATASET_PATH + file)
                json.dump(data, open(TARGET_PATH + file, "w"))
            except:
                continue
