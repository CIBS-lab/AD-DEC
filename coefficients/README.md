# Risk-equation coefficients

The cardiovascular scoring code (`../prevent.py`, `../pce.py`) separates the
equation form from the numeric coefficients held here, so that the source and
version of every coefficient are explicit and auditable. This is a deliberate
reproducibility choice, not an incomplete state.

To activate the scoring:

1. `prevent_totalcvd_10yr.json`: fill each null from the Khan et al. (2024)
   Circulation supplemental appendix (10-year total CVD, base model), then set
   `_verified` to true.
2. `pce_ascvd_10yr.json`: fill `S0`, `mean_S`, and each `beta` for the four groups
   from Goff et al. (2014) Appendix 7, then set `_verified` to true.
3. Confirm each assembled function against the official calculator (AHA PREVENT;
   a reference PCE calculator) for several test participants.

Until a file is verified, its module raises a clear error rather than returning an
unverified risk. Pasting the lab's validated coefficients reproduces the manuscript
numbers.
