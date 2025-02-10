import sys
sys.path.insert(0, "/home/lxt/EyesNet/GSAS-II/GSASII")
import GSASIIscriptable as G2sc
from monty.serialization import loadfn
import json
from icecream import ic
from tqdm import tqdm
import os

GPX_PATH = "gen_data.gpx"
PRM_PATH = "CuKa_lab_data.instprm"
CIF_PATH = "gen_data.cif"
DOCS_PATH = "/home/lxt/EyesNet/dataset/materials_project/total-materials.json.gz"
TARGET_PATH = "/home/lxt/EyesNet/dataset/xrd_simulation_data/XRD_total"

dataname = "CW x-ray simulation"
Tmin = 10
Tmax = 80
Tstep = 0.02

def gen_xrd():
    gpx = G2sc.G2Project(newgpx=GPX_PATH)
    gpx.set_Controls("cycles", 0)

    hist = gpx.add_simulated_powder_histogram(
        histname=dataname,
        iparams=PRM_PATH,
        Tmin=Tmin,
        Tmax=Tmax,
        Tstep=Tstep,
    )
    phase = gpx.add_phase(CIF_PATH, histograms=[hist])
    gpx.do_refinements()

    xrd_data = gpx.histogram(0).data["data"][1].data[1]

    return xrd_data

import os

def get_file_names(folder_path):
    file_names = []
    for file in os.listdir(folder_path):
        file_name = os.path.splitext(file)[0]
        file_names.append(file_name)
    return file_names

if __name__ == "__main__":
    file_names = get_file_names(TARGET_PATH)
    ic("start loding docs...")
    docs = loadfn(DOCS_PATH)
    ic("docs loaded")
    
    for doc in tqdm(docs):
        try:
            material_id = doc["material_id"]
            if material_id in file_names:
                continue
            formula_pretty = doc["formula_pretty"]
            structure = doc["structure"]
            cif_info = structure.to(CIF_PATH)
            XRD_simulation = gen_xrd()
        
            info = {
                "material_id": material_id,
                "formula_pretty": formula_pretty,
                "structure":structure.to_json(),
                "CIF_info": cif_info,
                "XRD_simulation": XRD_simulation.tolist(),
            }

            json.dump(info, open(f"{TARGET_PATH}/{material_id}.json", "w"))
        except:
            continue