"""
File containing the illustrative number needed to treat (NNT) for cardiovascular
prevention, by DEC phenotype (manuscript Supplementary Table S15).

NNT is computed per phenotype as 1 / (median PREVENT 10-year total CVD risk times an
assumed relative risk reduction). These figures are illustrative and literature
anchored; they are not estimated from HABS-HD, which has no intervention arm. Number
needed to harm is treated as indication-independent and roughly constant (on the
order of 1 excess case per 100 to 250 treated for statin-associated new-onset
diabetes) and is therefore not phenotype-specific.
"""
import csv
import sys

# Median PREVENT 10-year total CVD risk per phenotype (percent), from the paper.
# Replace with values recomputed by prevent.score_frame on HABS-HD if desired.
MEDIAN_RISK_PCT = {
    "C3 High-risk metabolic": 18.4,
    "C2 Vascular-neurodegeneration": 14.9,
    "C4 Amyloid-tau-glial": 14.0,
    "C5 Metabolic-amyloid": 10.0,
    "C1 Resilient": 7.4,
    "C6 Younger-tau": 6.0,
}

RRRS = (0.20, 0.25, 0.30)


def nnt(risk_fraction, rrr):
    """
    Number needed to treat for a given baseline risk and relative risk reduction.

    Parameters
    ----------
    risk_fraction : float
        Baseline 10-year risk in the range 0 to 1.
    rrr : float
        Relative risk reduction in the range 0 to 1.

    Returns
    -------
    nnt : float
        Number needed to treat.
    """
    arr = risk_fraction * rrr
    return 1.0 / arr if arr > 0 else float("inf")


def nnt_table(median_risk_pct=None, rrrs=RRRS):
    """
    Build the illustrative NNT table by phenotype.

    Parameters
    ----------
    median_risk_pct : dict, optional
        Phenotype to median 10-year risk (percent). The default is MEDIAN_RISK_PCT.
    rrrs : tuple of float, optional
        Relative risk reductions to tabulate. The default is RRRS.

    Returns
    -------
    rows : list of dict
        One row per phenotype with the NNT at each relative risk reduction.
    """
    median_risk_pct = median_risk_pct or MEDIAN_RISK_PCT
    rows = []
    for name, pct in median_risk_pct.items():
        r = pct / 100.0
        rows.append({"phenotype": name, "median_10y_risk_pct": pct,
                     **{"NNT_RRR_{}pct".format(int(x * 100)): round(nnt(r, x)) for x in rrrs}})
    return rows


if __name__ == "__main__":
    rows = nnt_table()
    cols = ["phenotype", "median_10y_risk_pct"] + ["NNT_RRR_{}pct".format(int(x * 100)) for x in RRRS]
    w = csv.DictWriter(sys.stdout, fieldnames=cols)
    w.writeheader()
    for row in rows:
        w.writerow(row)
