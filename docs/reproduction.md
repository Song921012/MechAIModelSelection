# Reproduction checklist

1. Install the repository and the tagged Python package dependency.
2. Run `python -m mechai_experiments.analyze --audit`.
3. Rebuild figures and tables from the released records.
4. Use the smoke profile for an end-to-end installation check.
5. Run the submission profile only when intentionally repeating the fits.

The figure script exports PDF, SVG, PNG, and TIFF locally. TIFF files are
excluded from Git because the 600-dpi bundle is several hundred megabytes.
