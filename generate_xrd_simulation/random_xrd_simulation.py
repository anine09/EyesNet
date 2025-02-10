import sys

sys.path.insert(0, "/home/lxt/EyesNet/GSAS-II/GSASII")
import GSASIIscriptable as G2sc

G2sc.SetPrintLevel("none")
import ujson
from tqdm import tqdm
import os

sys.path.insert(0, "/home/lxt/EyesNet/eyesnet")
from agent.random_xrd import random_xrd


GPX_PATH = "gen_data.gpx"
PRM_PATH = "CuKa_lab_data.instprm"
CIF_PATH = "gen_data.cif"

dataname = "eyesnet"
Tmin = 10
Tmax = 80
Tstep = 0.02

DATASET_PATH = "/home/lxt/EyesNet/dataset/crystal_graph/XRD_total/"
datafiles = os.listdir(DATASET_PATH)


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
    phase = gpx.add_phase(CIF_PATH, phasename="eyesnet", histograms=[hist])
    random_xrd(gpx)
    gpx.do_refinements()

    xrd_data = gpx.histogram(0).data["data"][1].data[1]

    return xrd_data

for file in tqdm(datafiles):
    random_XRD_simulation = []
    with open(DATASET_PATH + file, "r") as js_f:
        data = ujson.load(js_f)
    if "random_XRD_simulation" in data:
        continue
    with open("gen_data.cif", "w") as f:
        f.write(data["CIF_info"])
    for _ in range(5):
        random_XRD_simulation.append(gen_xrd().tolist())
    data["random_XRD_simulation"] = random_XRD_simulation
    with open(DATASET_PATH + file, "w") as js_f:
        ujson.dump(data, js_f)