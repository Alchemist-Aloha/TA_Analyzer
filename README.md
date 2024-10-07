# TA_Analyzer
Jupyter notebook for transient absorption analysis. Import 2d transient absorption matrix. Do background and zero time correction. Convert the matrix to Glotaran or Pyglotaran input. 

# Features
1. Compare and average multiple parallel TA experiments. e.g. Plotting multiple TA traces to verify photo damage or power dependence. 
2. Load TA matrix and do background correction.
3. Zerotime correction with hand draw line (using correction_line.py)
4. Fit the matrix and get the zerotime correction automatically. This feature only works for well behaved tamtrix (clean initial riseup without oscillation) otherwise the result will be not as good as hand draw line.
5. Get TA spectra at multiple time points and TA kinetics at certain time point.
6. Fit the TA kinetic traces with multiexponential using at most 4 time constants. The initial instrument response function is fit with a gaussian CDF function.
7. Output the corrected ta matrix to glotaran input format. Use tamatrix_importer.glotaran() for Glotaran Legacy (.ascii file for old Java version). Use tamatrix_importer.pyglotaran() for pyglotaran (xarray dataset input)

# To do
1. Better output of the fitting results. Currently copying the result manually is required.
2. background subtraction with blank matrix.
3. Add optional weight mask to kinetic trace fitting.
4. Possibly a GUI version in the future.
