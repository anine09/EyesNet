import orjson
from scandir import scandir


import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.nn.functional import pad
from torch_geometric.data import Data


class EyesNetDataset(Dataset):

    def __init__(self, dataset_path, round_=True, ICSD=True, scale=10000):
        self.xrd_data_list = []
        self.crystal_graph_list = []
        index = 0

        if ICSD:
            self.dataset_path = dataset_path + "/XRD_exp/"
        else:
            self.dataset_path = dataset_path + "/XRD_total/"
        for entry in tqdm(scandir(self.dataset_path)):
            # if index > 1000:
            #     break
            # index += 1
            with open(self.dataset_path + entry.name, "rb") as f:
                data = orjson.loads(f.read())
            xrd_data = torch.tensor(data["XRD_simulation"])
            if round_:
                xrd_data = torch.abs(torch.round(xrd_data))
            

            if scale:
                min_val = xrd_data.min()
                max_val = xrd_data.max()
                xrd_data = (xrd_data - min_val) / (max_val - min_val) * scale
            
            xrd_data = xrd_data.int()
            self.xrd_data_list.append(xrd_data)

            atomic_numbers = data["crystal_graph"]["atomic_numbers"]
            pos = data["crystal_graph"]["pos"]

            self.crystal_graph_list.append(
                Data(
                    atomic_numbers=torch.tensor(atomic_numbers),
                    pos=torch.tensor(pos),
                )
            )

    def __len__(self):
        return len(self.xrd_data_list)

    def __getitem__(self, idx):
        return (
            self.xrd_data_list[idx],
            self.crystal_graph_list[idx],
        )
