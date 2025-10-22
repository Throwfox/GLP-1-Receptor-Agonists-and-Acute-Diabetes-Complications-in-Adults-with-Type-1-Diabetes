# CATE Estimation Pipeline for T1D Study

## Overview

This pipeline implements Conditional Average Treatment Effect (CATE) estimation using the DRLearner (Doubly Robust Learner) method with bootstrap hyperparameter tuning.

## Key Features

### 1. Three-Block Cross-Fitting Design
- Data is divided into three blocks, stratified by both treatment and outcome
- Each individual's ITE is estimated using models trained on data that **excludes** them
- This ensures unbiased estimation and prevents overfitting

### 2. Bootstrap Hyperparameter Tuning
- **100 bootstrap iterations** for robust hyperparameter selection
- Each iteration uses **stratified 70/30 split** (training/validation)
- Separate tuning for propensity score model and outcome model
- Hyperparameters tested:
  - `n_estimators`: [50, 100, 150]
  - `learning_rate`: [0.01, 0.05, 0.1, 0.3]
  - `colsample_bytree`: [0.6, 0.8, 1.0]
  - `max_depth`: [3, 5, 7]

### 3. DRLearner with Default CV=3
- Uses EconML's DRLearner implementation
- Default 3-fold cross-validation (cv=3) for final model fitting
- Combines propensity score weighting with outcome regression

### 4. Workflow

```
For each outcome:
  1. Split data into 3 blocks (stratified by treatment × outcome)
  
  2. For each block (rotation):
     a. Use other 2 blocks as training data
     b. Bootstrap hyperparameter tuning (100 iterations)
        - Tune propensity score model
        - Tune outcome model
     c. Train DRLearner with optimal hyperparameters (cv=3)
     d. Estimate ITE on held-out block
  
  3. Combine ITE estimates from all 3 rotations
     → Calculate ATE and CI from rotation-based ITE
     → These are your PRIMARY RESULTS
  
  4. [OPTIONAL] Train final model on all data for interpretation:
     - Bootstrap hyperparameter tuning on all data
     - Train DRLearner (cv=3)
     - Generate interpretable decision tree
     - NOTE: This is ONLY for visualization, NOT for ITE estimation
  
  5. Save results to CSV (using rotation-based ITE)
```

**Important**: ITE estimates always come from rotation (step 3). The final model (step 4) is only for generating interpretable visualizations.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python test.py
```

Make sure `FINAL_DATA_T1D_03252025.csv` is in the same directory.

### Configuration Options

Edit `test.py` to customize:

```python
# At the bottom of test.py
TRAIN_FINAL_MODEL = True   # Generate interpretability visualizations
# TRAIN_FINAL_MODEL = False  # Skip final model (saves ~50% time)
```

Set to `False` if you only need ITE estimates without visualization.

## Outcomes Analyzed

1. Hospitalization or ED visit
2. Diabetic Ketoacidosis
3. Severe Hypoglycemia
4. Hyperglycemic Hyperosmolar State

## Output Files

For each outcome, the pipeline generates:
- `CATE_results_for_{outcome}.csv`: Contains individual-level CATE estimates
- Interpretable decision tree visualization (displayed during execution)

## Output CSV Columns

- All baseline covariates (after dummy encoding)
- `outcome`: Binary outcome indicator (0/1)
- `treatment`: Binary treatment indicator (0/1)
- `CATE`: Conditional Average Treatment Effect (individual-level estimate)

## Method Details

### Why This Approach?

1. **Three-Block Rotation**: Ensures each individual's treatment effect is estimated using models that were **not** trained on their data, preventing data leakage and overfitting.

2. **Bootstrap Hyperparameter Tuning**: More robust than single train/validation split, especially important for small sample sizes or imbalanced outcomes.

3. **Stratified Splits**: Maintains treatment and outcome proportions across blocks and bootstrap samples.

4. **DRLearner**: Doubly robust method that combines the strengths of:
   - Propensity score weighting
   - Outcome regression
   - Provides valid estimates if either model is correctly specified

5. **Default CV=3**: DRLearner internally uses 3-fold cross-validation for nuisance parameter estimation, providing additional robustness.

## Key References

- Kennedy, E. H. (2023). "Towards optimal doubly robust estimation of heterogeneous causal effects." Electronic Journal of Statistics.
- Chernozhukov, V., et al. (2018). "Double/debiased machine learning for treatment and structural parameters." The Econometrics Journal.

## Important Notes

### Propensity Score Handling

**This implementation does NOT pre-compute propensity scores as features.**

Unlike some notebook implementations, `test.py`:
- ✅ Does NOT use `LogisticRegression` to pre-calculate propensity scores
- ✅ Does NOT add propensity scores to the feature matrix `X`
- ✅ Lets `DRLearner` estimate propensity scores internally using `XGBClassifier`

**Why?**
1. Pre-computing PS and adding it to X creates circularity (information leakage)
2. DRLearner's internal cross-fitting (cv=3) requires PS estimation on independent folds
3. Using XGBoost for PS (instead of Logistic Regression) captures non-linear treatment assignment

See `DIFFERENCES_FROM_NOTEBOOK.md` for detailed explanation.

### Other Technical Details

- Random seed is set to 882046 for reproducibility
- Missing values in numeric covariates are imputed with column means
- Categorical variables are one-hot encoded (drop_first=False)
- No propensity score clipping needed (handled internally by DRLearner)

## License

This code is part of the iScience T1D research project.

## Contact

For questions or issues, please contact the research team.

