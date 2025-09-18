# eye_movement

This repository contains an end-to-end Jupyter notebook and supporting artifacts for building and evaluating desktop activity recognition models from eye-tracking data.

Contents
--------
- `eye_tracking_analysis.ipynb` — the main Jupyter notebook. It implements data loading, preprocessing, sliding-window feature extraction, event detection (I‑DT — fixation/saccade), model training and comparison (RandomForest and XGBoost), class-imbalance handling (SMOTE), hyperparameter tuning (GridSearchCV for RandomForest), and visualization cells that compare model performance.
- `dataset/` — CSV files for each recording (e.g., `P01_READ.csv`) used by the notebook. The notebook expects `x` and `y` columns at minimum; a `timestamp` column is supported if present.
- `rf_eye_tracking.joblib`, `xgb_eye_tracking.joblib` — example saved model artifacts (may be created after running notebook cells).
- `eye_tracking_analysis.py` — companion script (if present) with helper routines.

Quick start
-----------
1. Create and activate a Python virtual environment (recommended):

	# Windows (PowerShell)
	python -m venv .venv
	.\.venv\Scripts\Activate.ps1

2. Install dependencies (the notebook uses scikit-learn, pandas, numpy, matplotlib, seaborn, xgboost, imbalanced-learn, joblib):

	pip install -r requirements.txt

3. Start Jupyter Lab or Notebook and open `eye_tracking_analysis.ipynb`:

	jupyter lab

4. Run notebook cells in order. Recommended order:
	- Run the imports & constants cell.
	- Run the data loading cell (`load_all_data`) to produce `df_raw`.
	- Run preprocessing (velocity) and feature extraction cells that create `X_df`, `y` and `feature_names`.
	- Run the encoding/train cell (RandomForest baseline).
	- Optionally run SMOTE and the comparison cell to compare `RandomForest` and `XGBoost` with/without SMOTE.
	- Run the visualization cells to produce accuracy bar charts, confusion matrices, and feature importances.

What the notebook does (summary)
-------------------------------
- Data ingestion: reads CSVs from `dataset/`, extracts `user` and `task` from filenames, and forms session ids.
- Preprocessing: computes per-session differences (`dx`, `dy`) and a `velocity` feature (handles optional `timestamp` column). Velocity is clipped to remove extreme outliers.
- Event detection (I‑DT): detects fixations and saccades using a dispersion-threshold identification algorithm; events are used to create event-based window features (e.g., fixation count, mean fixation duration, saccade amplitude).
- Sliding-window features: converts sessions into overlapping windows (configurable `window_size` and `step_size`) and computes statistical features (means, stds, min/max) for `x`, `y`, and `velocity` per window.
- Modeling: trains a `RandomForestClassifier` baseline; includes a cell that trains an `XGBoost` classifier for comparison.
- Class imbalance: includes a SMOTE cell that oversamples only the training set and prints distributions before/after resampling.
- Model comparison & tuning: comparison cell trains RF and XGB on original vs SMOTE-resampled training data and reports accuracies and classification reports; a GridSearchCV cell is provided to tune RandomForest hyperparameters.
- Visualization: accuracy bar plots, normalized confusion matrices, and top-feature importance plots are included with an added human-readable summary and interpretation tips.

Notes & troubleshooting
-----------------------
- Run cells top-to-bottom to avoid NameError from missing variables (cells create and rely on globals like `X_df`, `y`, `X_train`, `X_test`).
- If you edit the notebook structure (add/remove cells), re-run the important preprocessing cells to re-create `X_df` and `y` before training cells.
- If `imbalanced-learn` or `xgboost` are missing, install them with `pip install imbalanced-learn xgboost`.

Next steps you can try
---------------------
- Tune XGBoost with GridSearchCV or use early-stopping with a validation split.
- Experiment with different SMOTE ratios, `window_size` and `step_size`, and I‑DT thresholds for event detection.
- Export the best model and build a small CLI script in `eye_tracking_analysis.py` to run inference on new recordings.

License
-------
See `LICENSE` in the repository root.

Contact
-------
This project was developed in this workspace. If you want changes (move markdown, add automated text summaries under plots, or export plots to PNG), tell me which change and I will update the notebook.
