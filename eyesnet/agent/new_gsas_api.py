import sys

sys.path.insert(0, "/home/lxt/EyesNet/GSAS-II/GSASII")
import GSASIIscriptable as G2sc

from fastapi import FastAPI
from pydantic import BaseModel
import optuna
from typing import List
import os
import shutil
from functools import partial
import uuid
import io
from contextlib import contextmanager
from copy import deepcopy

app = FastAPI()
project = {}


@contextmanager
def capture_stdout():
    output_capture = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = output_capture
        yield output_capture
    finally:
        sys.stdout = old_stdout
        output_capture.close()


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
        "instrument_zero": {"Instrument Parameters": ["Zero"]},
        "instrument_SH/L": {"Instrument Parameters": ["SH/L"]},
        "instrument_Polariz.": {"Instrument Parameters": ["Polariz."]},
        "unit_cell": {"Cell": True},
        "cell_size": {"Size": {"refine": True}},
        "cell_microstrain": {"Mustrain": {"refine": True}},
        "atom_X": {"Atoms": {lable: "X" for lable in project[proj_id]["ATOMS"]}},
        "atom_U": {"Atoms": {lable: "U" for lable in project[proj_id]["ATOMS"]}},
        "atom_F": {"Atoms": {lable: "F" for lable in project[proj_id]["ATOMS"]}},
    }


def background_fit_func(trial, proj_id):
    refine_setting = {
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
    proj = deepcopy(project[proj_id])
    proj["gpx"].do_refinements([refine_setting])
    score = proj["hist"].get_wR()
    del proj
    return score


def get_refinements(proj_id, actions: list):
    set_dict = {"once": {}}
    for action in actions:
        set_dict["once"].update(project[proj_id]["refine_set_map"][action])
    return [set_dict]


class FilePath(BaseModel):
    powder_data_path: str
    inst_parmas_path: str
    cif_path: str
    output_dir: str


class RefineData(BaseModel):
    actions: List[str]
    proj_id: str


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
        f"{file_path.output_dir}/rl_learn/blank.gpx",
        file_path.powder_data_path,
        file_path.inst_parmas_path,
        file_path.cif_path,
    )
    proj_id = str(uuid.uuid4())
    # proj_id = "test"
    project[proj_id] = {}
    running_dir = file_path.output_dir + "/rl_learn"
    gpx.filename = f"{running_dir}/{proj_id}/eyesnet.gpx"

    project[proj_id]["gpx"] = gpx
    project[proj_id]["hist"] = hist
    project[proj_id]["phase"] = phase

    if not (os.path.exists(data_path := f"{running_dir}/{proj_id}")):
        os.makedirs(data_path)

    project[proj_id]["output_dir"] = running_dir

    project[proj_id]["ATOMS"] = [atom.label for atom in phase.atoms()]

    project[proj_id]["refine_set_map"] = gen_refine_set_map(proj_id)
    return proj_id


@app.post("/fit_background")
def fit_background(proj_id: str, study_epoch: int):
    bg_fit_func = partial(background_fit_func, proj_id=proj_id)
    study = optuna.create_study(direction="minimize")
    study.optimize(bg_fit_func, n_trials=study_epoch, show_progress_bar=True)
    refine_setting = {
        "once": {
            "Background": {
                "type": study.best_params["background_func"],
                "refine": True,
                "no. coeffs": study.best_params["no_coeffs"],
            }
        }
    }
    project[proj_id]["gpx"].do_refinements([refine_setting])
    wR = project[proj_id]["hist"].get_wR()
    return {"wR": wR, **study.best_params}


@app.get("/get_Ycalc")
def get_Ycalc(proj_id: str):
    Ycalc = project[proj_id]["hist"].getdata("Ycalc").data
    return {"Ycalc": Ycalc.tolist()}


@app.get("/get_Yobs")
def get_Yobs(proj_id: str):
    Yobs = project[proj_id]["hist"].getdata("Yobs").data
    return {"Yobs": Yobs.tolist()}


@app.get("/get_wR")
def get_wR(proj_id: str):
    wR = project[proj_id]["hist"].get_wR()
    return {"wR": wR}


@app.post("/do_refinements")
def do_refinements(refine_data: RefineData):
    refinements = get_refinements(refine_data.proj_id, refine_data.actions)

    with capture_stdout() as captured:
        project[refine_data.proj_id]["gpx"].do_refinements(refinements)
        captured_output = captured.getvalue()
    if "***** Refinement successful *****" in captured_output:
        is_success = True
    else:
        is_success = False
    print(captured_output)
    Ycalc = project[refine_data.proj_id]["hist"].getdata("Ycalc").data
    wR = project[refine_data.proj_id]["hist"].get_wR()
    return {"wR": wR, "Ycalc": Ycalc.tolist(), "is_success": is_success}


@app.get("/save_project")
def save_project(proj_id: str):
    save_path = f"{project[proj_id]['gpx'].filename.replace('.gpx', '')}_best.gpx"
    project[proj_id]["gpx"].save(save_path)
    return f"success to save gpx file at: {save_path}"


@app.get("/remove_project")
def remove_project(proj_id: str):
    remove_path = f"{project[proj_id]['output_dir']}/{proj_id}"
    if os.path.exists(remove_path):
        shutil.rmtree(remove_path)
    del project[proj_id]
    return f"success to remove project: {proj_id}"
