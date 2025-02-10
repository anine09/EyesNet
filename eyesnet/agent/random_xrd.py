import random
import numpy as np
import pandas as pd
import math


def calculate_lattice_volume(a, b, c, alpha, beta, gamma):
    """
    计算晶格体积

    参数:
    a, b, c: 晶格参数 (单位：Å)
    alpha, beta, gamma: 晶格角度 (单位：度)

    返回:
    晶格体积 (单位：Å^3)
    """
    # 将角度转换为弧度
    alpha_rad = math.radians(alpha)
    beta_rad = math.radians(beta)
    gamma_rad = math.radians(gamma)

    # 计算体积
    volume = (
        a
        * b
        * c
        * math.sqrt(
            1
            + 2 * math.cos(alpha_rad) * math.cos(beta_rad) * math.cos(gamma_rad)
            - math.cos(alpha_rad) ** 2
            - math.cos(beta_rad) ** 2
            - math.cos(gamma_rad) ** 2
        )
    )

    return volume


def add_random_delta(n, a, b):
    return n + np.random.uniform(a, b, size=len(n))


def random_atoms_data(df, mode):
    num_samples = random.randint(1, df.shape[0])
    random_indices = np.random.choice(df.index, num_samples, replace=False)

    if "coor" in mode:
        df.loc[random_indices, ["x", "y", "z"]] = df.loc[
            random_indices, ["x", "y", "z"]
        ].apply(add_random_delta, args=(0.0, 1e-3))
    if "frac" in mode:
        df.loc[random_indices, ["frac"]] = df.loc[random_indices, ["frac"]].apply(
            add_random_delta, args=(0.0, 0.5)
        )
    if "Uiso" in mode:
        df.loc[random_indices, ["Uiso"]] = df.loc[random_indices, ["Uiso"]].apply(
            add_random_delta, args=(0.0, 0.05)
        )

    return df


def random_Cell(gpx):
    cell = gpx.data["Phases"]["eyesnet"]["General"]["Cell"][1:7]
    cell += np.random.uniform(
        0.0, 0.1, size=len(cell)
    )  # 所有晶胞参数（a, b, c, alpha, beta, gamma）在 [0, 0.1) Å 范围随机波动
    v = calculate_lattice_volume(*cell)  # 计算晶胞体积，GSAS-II 会自动重新计算晶胞密度
    gpx.data["Phases"]["eyesnet"]["General"]["Cell"][1:] = cell.tolist() + [v]


def random_Size(gpx):
    phase_data = gpx.data["Phases"]["eyesnet"]["Histograms"]["PWDR eyesnet"]
    phase_data["Size"][1][1] = random.uniform(
        0.5, 1
    )  # 设置在 [0.5, 1) 的随机晶粒尺寸峰值展宽的 Scherrer 常数


def random_Mustrain(gpx):
    phase_data = gpx.data["Phases"]["eyesnet"]["Histograms"]["PWDR eyesnet"]
    phase_data["Mustrain"][1][1] = random.uniform(
        0.0, 1000.0
    )  # 设置在 [0, 1000) 的随机微应变峰值展宽


def random_Atoms(gpx):
    phase_atoms = gpx.data["Phases"]["eyesnet"]["Atoms"]
    phase_atoms = pd.DataFrame(
        phase_atoms,
        columns=[
            "Name",
            "Type",
            "refine",
            "x",
            "y",
            "z",
            "frac",
            "site sym",
            "mult",
            "I/A",
            "Uiso",
            *([None] * (len(phase_atoms[0]) - 11)),
        ],
    )
    mode_num = random.randint(0, 3)
    mode = random.sample(["coor", "frac", "Uiso"], mode_num)
    gpx.data["Phases"]["eyesnet"]["Atoms"] = (
        random_atoms_data(phase_atoms, mode).to_numpy().tolist()
    )


def random_instrument(gpx):
    instrument_parameters = gpx.data["PWDR eyesnet"]["Instrument Parameters"][0]
    mode_list = [
        "Zero",
        "Polariz.",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "SH/L",
    ]
    mode_num = random.randint(0, len(mode_list))
    mode = random.sample(mode_list, mode_num)

    if "Zero" in mode:
        instrument_parameters["Zero"][1] = random.uniform(-10.0, 10.0)

    if "Polariz." in mode:
        instrument_parameters["Polariz."][1] = random.uniform(0.0, 1.0)

    if "U" in mode:
        instrument_parameters["U"][1] = random.uniform(-4.0, 4.0)

    if "V" in mode:
        instrument_parameters["V"][1] = random.uniform(-4.0, 4.0)

    if "W" in mode:
        instrument_parameters["W"][1] = random.uniform(-10.0, 10.0)

    if "X" in mode:
        instrument_parameters["X"][1] = random.uniform(-1.0, 1.0)

    if "Y" in mode:
        instrument_parameters["Y"][1] = random.uniform(-1.0, 1.0)

    if "Z" in mode:
        instrument_parameters["Z"][1] = random.uniform(-1.0, 1.0)

    if "SH/L" in mode:
        instrument_parameters["SH/L"][1] = random.uniform(0.0, 0.01)


def random_xrd(gpx):
    mode_list = [
        "Cell",
        "Size",
        "Mustrain",
        "Atoms",
        "Instrument",
    ]
    mode_num = random.randint(0, len(mode_list))
    mode = random.sample(mode_list, mode_num)

    if "Cell" in mode:
        random_Cell(gpx)

    if "Size" in mode:
        random_Size(gpx)

    if "Mustrain" in mode:
        random_Mustrain(gpx)

    if "Atoms" in mode:
        random_Atoms(gpx)

    if "Instrument" in mode:
        random_instrument(gpx)
