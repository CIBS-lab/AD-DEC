"""
File containing the 2013 ACC/AHA Pooled Cohort Equations (PCE) for 10-year
atherosclerotic cardiovascular disease (ASCVD) risk. Used only for the sensitivity
comparison in the manuscript (Supplementary Table S16), where re-scoring with the
race-based PCE reintroduces a race coefficient that the race-free PREVENT removes.

Reference: Goff DC Jr, et al. Circulation. 2014;129(25 Suppl 2):S49-S73.

As with prevent.py, the equation form is here and the four coefficient sets are held
in coefficients/pce_ascvd_10yr.json with explicit provenance. The PCE has no equation
for Hispanic adults; they are scored with the white_or_other coefficients by
convention, a stated limitation.
"""
import json
import math
import os

COEF_PATH = os.path.join(os.path.dirname(__file__), "coefficients", "pce_ascvd_10yr.json")


def load_coefficients(path=COEF_PATH):
    """
    Load and validate the PCE coefficient file.

    Parameters
    ----------
    path : str, optional
        Path to the coefficient JSON. The default is COEF_PATH.

    Returns
    -------
    coef : dict
        Dictionary with the four PCE group coefficient sets.
    """
    with open(path) as f:
        coef = json.load(f)
    if not coef.get("_verified", False):
        raise RuntimeError(
            os.path.basename(path) + " is not verified. Populate from Goff et al. "
            "(2014), confirm against a reference calculator, and set \"_verified\" to true.")
    return coef


def pce_group_key(row):
    """
    Return the PCE coefficient-group key for a participant.

    Parameters
    ----------
    row : pandas.Series or mapping
        Must contain "black" and "female" flags.

    Returns
    -------
    key : str
        One of white_or_other_female, white_or_other_male, black_female, black_male.
    """
    return ("black_" if row["black"] else "white_or_other_") + ("female" if row["female"] else "male")


def pce_terms(row):
    """
    Compute the log-transformed PCE terms for one participant.

    Parameters
    ----------
    row : pandas.Series or mapping
        Must contain age, total_chol_mgdl, hdl_mgdl, sbp, treated_bp,
        current_smoker, diabetes.

    Returns
    -------
    terms : dict
        Transformed model terms.
    """
    ln_age = math.log(row["age"])
    ln_tc = math.log(row["total_chol_mgdl"])
    ln_hdl = math.log(row["hdl_mgdl"])
    ln_sbp = math.log(row["sbp"])
    treated = bool(row["treated_bp"])
    return {
        "ln_age": ln_age, "ln_age_sq": ln_age ** 2,
        "ln_tc": ln_tc, "ln_age_x_ln_tc": ln_age * ln_tc,
        "ln_hdl": ln_hdl, "ln_age_x_ln_hdl": ln_age * ln_hdl,
        "ln_sbp_treated": ln_sbp if treated else 0.0,
        "ln_age_x_ln_sbp_treated": (ln_age * ln_sbp) if treated else 0.0,
        "ln_sbp_untreated": ln_sbp if not treated else 0.0,
        "ln_age_x_ln_sbp_untreated": (ln_age * ln_sbp) if not treated else 0.0,
        "smoker": float(row["current_smoker"]),
        "ln_age_x_smoker": ln_age * float(row["current_smoker"]),
        "diabetes": float(row["diabetes"]),
    }


def pce_10y_ascvd(row, coef=None):
    """
    Compute the PCE 10-year ASCVD risk for one participant.

    Parameters
    ----------
    row : pandas.Series or mapping
        Participant row (see pce_terms and pce_group_key).
    coef : dict, optional
        Pre-loaded coefficients. Loaded from disk if None. The default is None.

    Returns
    -------
    risk : float
        10-year ASCVD risk in the range 0 to 1.
    """
    coef = coef or load_coefficients()
    g = coef[pce_group_key(row)]
    s = sum(g["beta"].get(k, 0.0) * v for k, v in pce_terms(row).items())
    return 1.0 - g["S0"] ** math.exp(s - g["mean_S"])


def score_frame(df, coef=None):
    """
    Score every participant in a DataFrame with the PCE.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns must include the PCE input fields.
    coef : dict, optional
        Pre-loaded coefficients. Loaded from disk if None. The default is None.

    Returns
    -------
    risk : pandas.Series
        10-year ASCVD risk per participant.
    """
    coef = coef or load_coefficients()
    return df.apply(lambda r: pce_10y_ascvd(r, coef), axis=1)


if __name__ == "__main__":
    try:
        load_coefficients()
        print("Coefficients verified. Import pce_10y_ascvd or score_frame to use.")
    except (RuntimeError, FileNotFoundError) as e:
        print(e)
