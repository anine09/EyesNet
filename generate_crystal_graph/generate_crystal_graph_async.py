import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice
from tqdm import tqdm
import orjson
from pymatgen.core import Structure

# 添加自定义模块路径
sys.path.insert(0, "/home/lxt/EyesNet/")
from eyesnet.reranker.crystal_encoder.Structure2Graph import (
    get_adjacency_matrix,
    get_atom_ids,
    get_coordinates,
)

# 更高效的文件遍历方法
from scandir import scandir

def generate_crystal_graph(file_path):
    """生成晶体图数据"""
    with open(file_path, "rb") as f:
        data = orjson.loads(f.read())
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

def process_file(file, DATASET_PATH, TARGET_PATH):
    """处理单个文件"""
    try:
        data = generate_crystal_graph(os.path.join(DATASET_PATH, file))
        with open(os.path.join(TARGET_PATH, file), "wb") as f:
            f.write(orjson.dumps(data))
        return True
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return False

def process_files_batch(files, DATASET_PATH, TARGET_PATH):
    """批量处理文件"""
    results = []
    for file in files:
        results.append(process_file(file, DATASET_PATH, TARGET_PATH))
    return results

def chunked_iterable(iterable, size):
    """将可迭代对象分块"""
    it = iter(iterable)
    return iter(lambda: list(islice(it, size)), [])

def get_files_to_process(DATASET_PATH, TARGET_PATH):
    """获取待处理的文件列表"""
    already_processed = set()
    for entry in scandir(TARGET_PATH):
        if entry.is_file():
            already_processed.add(entry.name)

    files_to_process = []
    for entry in scandir(DATASET_PATH):
        if entry.is_file() and entry.name not in already_processed:
            files_to_process.append(entry.name)

    return files_to_process

def main():
    DATASET_PATH = "/home/lxt/EyesNet/dataset/xrd_simulation_data/XRD_total/"
    TARGET_PATH = "/home/lxt/EyesNet/dataset/crystal_graph/XRD_total/"

    # 获取待处理的文件列表
    files_to_process = get_files_to_process(DATASET_PATH, TARGET_PATH)

    # 将文件分成每批 10 个
    batch_size = 10
    file_batches = chunked_iterable(files_to_process, batch_size)

    # 使用 ProcessPoolExecutor 并行处理文件
    with ProcessPoolExecutor() as executor:
        futures = []
        for batch in file_batches:
            futures.append(
                executor.submit(process_files_batch, batch, DATASET_PATH, TARGET_PATH)
            )

        # 使用 tqdm 显示进度
        with tqdm(total=len(files_to_process), desc="Processing files") as pbar:
            for future in as_completed(futures):
                future.result()  # 等待任务完成
                pbar.update(len(future.result()))

if __name__ == "__main__":
    main()