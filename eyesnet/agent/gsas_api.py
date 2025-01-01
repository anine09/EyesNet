import sys

sys.path.insert(0, "C:\Coding/EyesNet/gsasii/GSAS-II/GSASII")
import GSASIIscriptable as G2sc

# G2sc.SetPrintLevel("error")

from fastapi import FastAPI
from pydantic import BaseModel
import optuna
from typing import List
import os
import shutil
from functools import partial
import time
import random

app = FastAPI()
project = {}

BACKGROUND_FUNCTIONS = [
    "chebyshev",
    "chebyshev-1",
    "cosine",
    "Q^2 power series",
    "Q^-2 power series",
    "lin interpolate",
    "inv interpolate",
    "log interpolate",
]


def gen_refine_set_map(proj_id):
    return {
        "background_function": {"Background": {"refine": True}},
        "instrument_U": {"Instrument Parameters": ["U"]},
        "instrument_V": {"Instrument Parameters": ["V"]},
        "instrument_W": {"Instrument Parameters": ["W"]},
        "instrument_X": {"Instrument Parameters": ["X"]},
        "instrument_Y": {"Instrument Parameters": ["Y"]},
        "instrument_Z": {"Instrument Parameters": ["Z"]},
        "instrument_alpha": {"Instrument Parameters": ["alpha"]},
        "instrument_beta-0": {"Instrument Parameters": ["beta-0"]},
        "instrument_beta-1": {"Instrument Parameters": ["beta-1"]},
        "instrument_beta-q": {"Instrument Parameters": ["beta-q"]},
        "instrument_sig-0": {"Instrument Parameters": ["sig-0"]},
        "instrument_sig-1": {"Instrument Parameters": ["sig-1"]},
        "instrument_sig-2": {"Instrument Parameters": ["sig-2"]},
        "instrument_sig-q": {"Instrument Parameters": ["sig-q"]},
        "instrument_difA": {"Instrument Parameters": ["difA"]},
        "instrument_difB": {"Instrument Parameters": ["difB"]},
        "instrument_difC": {"Instrument Parameters": ["difC"]},
        "instrument_zero": {"Instrument Parameters": ["Zero"]},
        "instrument_SH/L": {"Instrument Parameters": ["SH/L"]},
        "instrument_Polariz.": {"Instrument Parameters": ["Polariz."]},
        "instrument_Lam": {"Instrument Parameters": ["Lam"]},
        "sigle_xtral_scale": {"Single xtal": ["Scale"]},
        "sigle_xtral_BabA": {"Single xtal": ["BabA"]},
        "sigle_xtral_BabU": {"Single xtal": ["BabU"]},
        "sigle_xtral_Eg": {"Single xtal": ["Eg"]},
        "sigle_xtral_Es": {"Single xtal": ["Es"]},
        "sigle_xtral_Ep": {"Single xtal": ["Ep"]},
        "sigle_xtral_Flack": {"Single xtal": ["Flack"]},
        "sample_displacement_y": {"Sample Parameters": ["DisplaceY"]},
        "sample_displacement_x": {"Sample Parameters": ["DisplaceX"]},
        "sample_absorption": {"Sample Parameters": ["Absorption"]},
        "sample_contrast": {"Sample Parameters": ["Contrast"]},
        "sample_scale": {"Sample Parameters": ["Scale"]},
        "unit_cell": {"Cell": True},
        "LeBail": {"LeBail": True},
        "cell_size": {"Size": {"refine": True}},
        "cell_microstrain": {"Mustrain": {"refine": True}},
        "HStrain": {"HStrain": True},
        "atom_X": {"Atoms": {lable: "X" for lable in project[proj_id]["ATOMS"]}},
        "atom_U": {"Atoms": {lable: "U" for lable in project[proj_id]["ATOMS"]}},
        "atom_F": {"Atoms": {lable: "F" for lable in project[proj_id]["ATOMS"]}},
        "Babinet_BabA": {"Babinet": ["BabA"]},
        "Babinet_BabU": {"Babinet": ["BabU"]},
        "Extinction": {"Extinction": True},
        "Pref.Ori.": {"Pref.Ori.": True},
        "Show": {"Show": True},
        "Use": {"Use": True},
        "Scale": {"Scale": True},
    }


def background_fit_func(trial, proj_id):
    pardict = {
        "once": {
            "Background": {
                "type": trial.suggest_categorical(
                    "background_func", BACKGROUND_FUNCTIONS
                ),
                "refine": True,
                "no. coeffs": trial.suggest_int("no_coeffs", 1, 36),
            }
        }
    }
    project[proj_id]["gpx"].do_refinements([pardict])
    score = project[proj_id]["hist"].get_wR()
    return score


def get_refinements(proj_id, actions: list):
    set_dict = {"once": {}}
    for action in actions:
        set_dict["once"].update(project[proj_id]["refine_set_map"][action])
    return [set_dict]


class FilePath(BaseModel):
    gpx_path: str
    powder_data_path: str
    inst_parmas_path: str
    cif_path: str
    data_dir: str


class RefineData(BaseModel):
    actions: List[str]
    proj_id: int


def create_gpx(
    gpx_name,
    powder_data_path,
    inst_parmas_path,
    cif_path,
):
    gpx = G2sc.G2Project(newgpx=gpx_name)
    hist = gpx.add_powder_histogram(powder_data_path, inst_parmas_path)
    phase = gpx.add_phase(cif_path, histograms=[hist])
    return gpx, hist, phase


@app.post("/create_project")
def create_project(file_path: FilePath):
    gpx, hist, phase = create_gpx(
        f"{file_path.data_dir}/rl_learn/blank.gpx",
        file_path.powder_data_path,
        file_path.inst_parmas_path,
        file_path.cif_path,
    )
    gpx.set_Controls('cycles', 10)
    proj_id = int(f"{id(gpx)}{int(time.time())}")
    project[proj_id] = {}
    running_dir = file_path.data_dir + "/rl_learn"
    gpx.filename = f"{running_dir}/{proj_id}/{file_path.gpx_path}"

    project[proj_id]["gpx"] = gpx
    project[proj_id]["hist"] = hist
    project[proj_id]["phase"] = phase
    project[proj_id]["error_path"] = gpx.filename.replace(".gpx", ".err")

    if not (os.path.exists(data_path := f"{running_dir}/{proj_id}")):
        os.makedirs(data_path)
    with open(project[proj_id]["error_path"], "w") as f:
        f.write("")
    project[proj_id]["data_dir"] = running_dir

    project[proj_id]["ATOMS"] = [atom.label for atom in phase.atoms()]

    project[proj_id]["refine_set_map"] = gen_refine_set_map(proj_id)

    return {"proj_id": proj_id, "message": "success to create project"}


@app.post("/fit_background")
def fit_background(proj_id: int, study_epoch: int):
    bg_fit_func = partial(background_fit_func, proj_id=proj_id)
    study = optuna.create_study(direction="minimize")
    study.optimize(bg_fit_func, n_trials=study_epoch, show_progress_bar=True)
    pardict = {
        "once": {
            "Background": {
                "type": study.best_params["background_func"],
                "refine": True,
                "no. coeffs": study.best_params["no_coeffs"],
            }
        }
    }
    project[proj_id]["gpx"].do_refinements([pardict])
    wR = project[proj_id]["hist"].get_wR()
    return {"wR": wR, **study.best_params}


@app.get("/get_residual")
def get_residuals(proj_id: int):
    residual = project[proj_id]["hist"].getdata("Residual").data
    return {"residual": residual.tolist()}


@app.get("/get_wR")
def get_wR(proj_id: int):
    wR = project[proj_id]["hist"].get_wR()
    return {"wR": wR}


@app.post("/do_refinements")
def do_refinements(refine_data: RefineData):
    refinements = get_refinements(refine_data.proj_id, refine_data.actions)
    project[refine_data.proj_id]["gpx"].do_refinements(refinements)
    with open(project[refine_data.proj_id]["error_path"], "r") as f:
        result = f.read()
        if "Refinement error" in result:
            truncation = True
        else:
            truncation = False

    residual = project[refine_data.proj_id]["hist"].getdata("Residual").data
    wR = project[refine_data.proj_id]["hist"].get_wR()
    return {"wR": wR, "residual": residual.tolist(), "truncation": truncation}


@app.get("/save_project")
def save_project(proj_id: int):
    save_path = f"{project[proj_id]['gpx'].filename.replace('.gpx', '')}_best.gpx"
    project[proj_id]["gpx"].save(save_path)
    return {"message": "success to save gpx file"}


@app.get("/remove_project")
def remove_project(proj_id: int):
    remove_path = f"{project[proj_id]['data_dir']}/{proj_id}"
    if os.path.exists(remove_path):
        shutil.rmtree(remove_path)
    del project[proj_id]
    return {"message": "success to remove project"}
