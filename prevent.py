"""
File containing the AHA PREVENT 10-year total cardiovascular disease (CVD) risk
score, used to place the WRAP-derived phenotypes on a guideline cardiovascular-risk
scale (manuscript Supplementary Tables S12 and S13). Sex-specific and race-free.

Reference: Khan SS, et al. Development and Validation of the American Heart
Association's PREVENT Equations. Circulation. 2024;149:430-449.

The equation form is implemented here; the coefficients are held separately in
coefficients/prevent_totalcvd_10yr.json so that their source and version are
explicit and auditable. Populate that file from the Khan et al. (2024) supplement,
verify against the official AHA PREVENT online calculator, and set "_verified" to
true. load_coefficients raises otherwise, so an unverified score is never returned.
"""
import json
import math
import os

MG_DL_TO_MMOL = 0.02586
COEF_PATH = os.path.join(os.path.dirname(__file__), "coefficients", "prevent_totalcvd_10yr.json")

REQUIRED = ("female", "age", "total_chol_mgdl", "hdl_mgdl", "sbp", "egfr",
            "diabetes", "current_smoker", "on_bp_meds", "on_statin")


def load_coefficients(path=COEF_PATH):
    """
    Load and validate the PREVENT coefficient file.

    Parameters
    ----------
    path : str, optional
        Path to the coefficient JSON. The default is COEF_PATH.

    Returns
    -------
    coef : dict
        Dictionary with "female" and "male" coefficient sets.
    """
    with open(path) as f:
        coef = json.load(f)
    if not coef.get("_verified", False):
        raise RuntimeError(
            os.path.basename(path) + " is not verified. Populate the coefficients "
            "from Khan et al. (2024), confirm against the AHA PREVENT calculator, "
            "and set \"_verified\" to true.")
    for sex in ("female", "male"):
        if any(v is None for v in coef[sex].values()):
            raise RuntimeError(path + ": '" + sex + "' still contains null coefficients.")
    return coef


def prevent_terms(row):
    """
    Compute the PREVENT base-model variable transforms for one participant.

    Parameters
    ----------
    row : pandas.Series or mapping
        Must contain the fields listed in REQUIRED.

    Returns
    -------
    terms : dict
        Transformed and centred model terms.
    """
    non_hdl = (row["total_chol_mgdl"] - row["hdl_mgdl"]) * MG_DL_TO_MMOL
    hdl = row["hdl_mgdl"] * MG_DL_TO_MMOL
    cage = (row["age"] - 55.0) / 10.0
    sbp_ge = (max(row["sbp"], 110.0) - 130.0) / 20.0
    return {
        "cage": cage,
        "non_hdl": non_hdl - 3.5,
        "hdl": (hdl - 1.3) / 0.3,
        "sbp_lt110": (min(row["sbp"], 110.0) - 110.0) / 20.0,
        "sbp_ge110": sbp_ge,
        "diabetes": float(row["diabetes"]),
        "smoker": float(row["current_smoker"]),
        "egfr_lt60": (min(row["egfr"], 60.0) - 60.0) / -15.0,
        "egfr_ge60": (max(row["egfr"], 60.0) - 90.0) / -15.0,
        "bp_meds": float(row["on_bp_meds"]),
        "statin": float(row["on_statin"]),
        "bpmeds_x_sbp_ge110": float(row["on_bp_meds"]) * sbp_ge,
        "statin_x_non_hdl": float(row["on_statin"]) * (non_hdl - 3.5),
        "cage_x_non_hdl": cage * (non_hdl - 3.5),
        "cage_x_hdl": cage * ((hdl - 1.3) / 0.3),
        "cage_x_sbp_ge110": cage * sbp_ge,
        "cage_x_diabetes": cage * float(row["diabetes"]),
        "cage_x_smoker": cage * float(row["current_smoker"]),
        "cage_x_egfr_lt60": cage * ((min(row["egfr"], 60.0) - 60.0) / -15.0),
    }


def prevent_10y_total_cvd(row, coef=None):
    """
    Compute the PREVENT 10-year total CVD risk for one participant.

    Parameters
    ----------
    row : pandas.Series or mapping
        Participant row containing the fields in REQUIRED.
    coef : dict, optional
        Pre-loaded coefficients. Loaded from disk if None. The default is None.

    Returns
    -------
    risk : float
        10-year total CVD risk in the range 0 to 1.
    """
    coef = coef or load_coefficients()
    c = coef["female" if row["female"] else "male"]
    logodds = c["intercept"] + sum(c[k] * v for k, v in prevent_terms(row).items() if k in c)
    return 1.0 / (1.0 + math.exp(-logodds))


def score_frame(df, coef=None):
    """
    Score every participant in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns must include the fields in REQUIRED.
    coef : dict, optional
        Pre-loaded coefficients. Loaded from disk if None. The default is None.

    Returns
    -------
    risk : pandas.Series
        10-year total CVD risk per participant.
    """
    coef = coef or load_coefficients()
    return df.apply(lambda r: prevent_10y_total_cvd(r, coef), axis=1)


if __name__ == "__main__":
    try:
        load_coefficients()
        print("Coefficients verified. Import prevent_10y_total_cvd or score_frame to use.")
    except (RuntimeError, FileNotFoundError) as e:
        print(e)
