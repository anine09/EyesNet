import orjson
from scandir import scandir


import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.nn.functional import pad


class EyesNetDataset(Dataset):

    def __init__(self, dataset_path, padding, round_=True, ICSD=True):
        self.xrd_data_list = []
        self.atom_ids_list = []
        self.coordinates_list = []
        self.adjacency_matrix_list = []
        index = 0
        
        if ICSD:
            self.dataset_path = dataset_path + "/XRD_exp/"
        else:
            self.dataset_path = dataset_path + "/XRD_total/"
        for entry in tqdm(scandir(self.dataset_path)):
            if index > 100:
                break
            index += 1
            with open(self.dataset_path + entry.name, "rb") as f:
                data = orjson.loads(f.read())
            xrd_data = torch.tensor(data["XRD_simulation"])
            if round_:
                xrd_data = torch.abs(torch.round(xrd_data)).int()
            self.xrd_data_list.append(xrd_data)

            atom_ids = data["crystal_graph"]["atom_ids"]
            coordinates = data["crystal_graph"]["coordinates"]
            adjacency_matrix = data["crystal_graph"]["adjacency_matrix"]

            # padding
            atom_ids = torch.tensor(atom_ids)
            atom_ids = pad(atom_ids, (0, padding - atom_ids.size(dim=0)), value=-1)

            coordinates = torch.tensor(coordinates)
            coordinates = pad(
                coordinates,
                (0, 0, 0, padding - coordinates.size(dim=0)),
                value=-1,
            )

            adjacency_matrix = torch.tensor(adjacency_matrix)
            padding_len = padding - adjacency_matrix.size(dim=0)
            adjacency_matrix = pad(
                adjacency_matrix, (0, padding_len, 0, padding_len), value=False
            )

            self.atom_ids_list.append(atom_ids)
            self.coordinates_list.append(coordinates)
            self.adjacency_matrix_list.append(adjacency_matrix)

    def __len__(self):
        return len(self.xrd_data_list)

    def __getitem__(self, idx):
        return (
            self.xrd_data_list[idx],
            self.atom_ids_list[idx],
            self.coordinates_list[idx],
            self.adjacency_matrix_list[idx],
        )
