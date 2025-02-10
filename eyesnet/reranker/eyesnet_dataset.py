import orjson
import torch
from scandir import scandir
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm


def xrd2tensor(xrd_data, round_=True, scale=10000):
    xrd_data = torch.tensor(xrd_data)
    if round_:
        xrd_data = torch.round(xrd_data)
    xrd_data = torch.abs(xrd_data)
    if scale:
        min_val = xrd_data.min()
        max_val = xrd_data.max()
        xrd_data = (xrd_data - min_val) / (max_val - min_val) * scale
    xrd_data = xrd_data.int()
    return xrd_data


class EyesNetDataset(Dataset):

    def __init__(self, dataset_path, round_=True, ICSD=True, random_=True, scale=10000):
        self.xrd_data_list = []
        self.crystal_graph_list = []
        # index = 0

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
    
            atomic_numbers = data["crystal_graph"]["atomic_numbers"]
            pos = data["crystal_graph"]["pos"]
            crystal_data = Data(
                    atomic_numbers=torch.tensor(atomic_numbers),
                    pos=torch.tensor(pos),
                )
            
            original_xrd_data = xrd2tensor(data["XRD_simulation"], round_=round_, scale=scale)
            self.xrd_data_list.append(original_xrd_data)
            self.crystal_graph_list.append(crystal_data)

            if random_:
                for random_xrd_data in data["random_XRD_simulation"]:
                    random_xrd_data = xrd2tensor(random_xrd_data, round_=round_, scale=scale)
                    self.xrd_data_list.append(random_xrd_data)
                    self.crystal_graph_list.append(crystal_data)

    def __len__(self):
        return len(self.xrd_data_list)

    def __getitem__(self, idx):
        return (
            self.xrd_data_list[idx],
            self.crystal_graph_list[idx],
        )
