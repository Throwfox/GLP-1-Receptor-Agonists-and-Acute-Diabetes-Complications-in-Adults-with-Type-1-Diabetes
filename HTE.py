"""
Complete Pipeline for CATE Estimation using DRLearner with Bootstrap Hyperparameter Tuning

This script implements:
1. Data divided into three blocks, stratified by treatment and outcome
2. Bootstrap hyperparameter tuning (100 iterations, 70/30 stratified split)
3. ITE estimation using models trained on data that excludes each individual
4. DRLearner with default cv=3 for cross-validation
"""

import numpy as np
import pandas as pd
import random
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import log_loss, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
from econml.dr import DRLearner
from econml.cate_interpreter import SingleTreeCateInterpreter
from collections import defaultdict
from itertools import product
from tqdm import tqdm, trange
from scipy import stats

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
SEED = 882046
random.seed(SEED)
np.random.seed(SEED)


def bootstrap_hyperparameter_tuning(X_train, y_train, T_train, model_type='propensity', 
                                     n_bootstrap=100, random_state=SEED):
    """
    Perform hyperparameter tuning using bootstrap with stratified 70/30 splits.
    
    Parameters:
    -----------
    X_train: numpy array, Training features
    y_train: numpy array, Training outcomes
    T_train: numpy array, Training treatment indicators
    model_type: str, 'propensity' or 'outcome'
    n_bootstrap: int, Number of bootstrap iterations (default: 100)
    random_state: int, Random seed
    
    Returns:
    --------
    best_params: dict, Best hyperparameters found
    """
    # Define hyperparameter grids
    if model_type == 'propensity':
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 5, 7]
        }
    else:  # outcome model
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1, 0.3],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 5, 7]
        }
    
    # Generate all parameter combinations
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                         for v in product(*param_grid.values())]
    
    # Store scores for each parameter combination
    param_scores = defaultdict(list)
    
    # Stratification: for propensity use T, for outcome use y
    stratify_var = T_train if model_type == 'propensity' else y_train
    
    # Bootstrap iterations with stratified split
    for bootstrap_iter in trange(n_bootstrap, desc=f"Bootstrap tuning ({model_type})"):
        # Stratified 70/30 split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, 
                                     random_state=random_state + bootstrap_iter)
        
        for train_idx, val_idx in sss.split(X_train, stratify_var):
            X_bt_train, X_bt_val = X_train[train_idx], X_train[val_idx]
            y_bt_train, y_bt_val = y_train[train_idx], y_train[val_idx]
            T_bt_train, T_bt_val = T_train[train_idx], T_train[val_idx]
            
            # Test each parameter combination
            for params in param_combinations:
                if model_type == 'propensity':
                    model = XGBClassifier(**params, random_state=random_state, 
                                        use_label_encoder=False, eval_metric='logloss')
                    model.fit(X_bt_train, T_bt_train)
                    y_pred_proba = model.predict_proba(X_bt_val)
                    score = log_loss(T_bt_val, y_pred_proba)  # Lower is better
                else:
                    model = XGBRegressor(**params, random_state=random_state)
                    model.fit(X_bt_train, y_bt_train)
                    y_pred = model.predict(X_bt_val)
                    score = mean_squared_error(y_bt_val, y_pred)  # Lower is better
                
                # Store the score
                param_key = tuple(sorted(params.items()))
                param_scores[param_key].append(score)
    
    # Find the parameter combination with the best average score
    best_params_tuple = min(param_scores.keys(), 
                           key=lambda k: np.mean(param_scores[k]))
    best_params = dict(best_params_tuple)
    
    print(f"\nBest {model_type} hyperparameters: {best_params}")
    print(f"Average validation score: {np.mean(param_scores[best_params_tuple]):.4f}")
    
    return best_params


def stratified_3fold_split(X, y, T, random_state=SEED):
    """
    Split data into 3 blocks stratified by both treatment and outcome.
    
    Parameters:
    -----------
    X: numpy array, Features
    y: numpy array, Outcomes
    T: numpy array, Treatment indicators
    random_state: int, Random seed
    
    Returns:
    --------
    blocks: list of tuples, Each tuple contains (X, y, T) for that block
    """
    # Create stratification variable combining treatment and outcome
    stratify_var = T * 2 + y
    
    # First split: 33.3% vs 66.7%
    X_temp, X_block3, y_temp, y_block3, T_temp, T_block3, strat_temp, strat_block3 = \
        train_test_split(X, y, T, stratify_var, test_size=1/3, 
                        random_state=random_state, stratify=stratify_var)
    
    # Second split: 50% of remaining (33.3% of total) vs 50% of remaining (33.3% of total)
    X_block1, X_block2, y_block1, y_block2, T_block1, T_block2 = \
        train_test_split(X_temp, y_temp, T_temp, test_size=0.5, 
                        random_state=random_state, stratify=strat_temp)
    
    blocks = [
        (X_block1, y_block1, T_block1),
        (X_block2, y_block2, T_block2),
        (X_block3, y_block3, T_block3)
    ]
    
    print(f"\nData split into 3 blocks:")
    for i, (X_b, y_b, T_b) in enumerate(blocks):
        print(f"  Block {i+1}: {len(X_b)} samples")
    
    return blocks


def estimate_cate_with_rotation(blocks, final_columns, random_state=SEED):
    """
    Estimate CATE using 3-fold rotation approach.
    Each individual's ITE is estimated using models trained on data that excludes them.
    
    Parameters:
    -----------
    blocks: list of tuples, Three data blocks
    final_columns: pandas Index, Column names for features
    random_state: int, Random seed
    
    Returns:
    --------
    all_ite: numpy array, Individual treatment effects for all samples
    all_indices: numpy array, Original indices for all samples
    """
    all_ite = []
    all_X = []
    all_y = []
    all_T = []
    
    # Rotate through 3 blocks
    for test_block_idx in range(3):
        print(f"\n{'='*60}")
        print(f"Round {test_block_idx + 1}: Testing on Block {test_block_idx + 1}")
        print(f"{'='*60}")
        
        # Get training blocks (the other two blocks)
        train_blocks = [i for i in range(3) if i != test_block_idx]
        
        # Combine training blocks
        X_train = np.vstack([blocks[i][0] for i in train_blocks])
        y_train = np.concatenate([blocks[i][1] for i in train_blocks])
        T_train = np.concatenate([blocks[i][2] for i in train_blocks])
        
        # Get test block
        X_test, y_test, T_test = blocks[test_block_idx]
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Step 1: Hyperparameter tuning on training blocks
        print("\nStep 1: Tuning propensity score model...")
        best_propensity_params = bootstrap_hyperparameter_tuning(
            X_train, y_train, T_train, 
            model_type='propensity', 
            n_bootstrap=100,
            random_state=random_state
        )
        
        print("\nStep 2: Tuning outcome model...")
        best_outcome_params = bootstrap_hyperparameter_tuning(
            X_train, y_train, T_train,
            model_type='outcome',
            n_bootstrap=100,
            random_state=random_state
        )
        
        # Step 2: Train DRLearner with optimal hyperparameters
        print("\nStep 3: Training DRLearner with optimal hyperparameters (cv=3)...")
        
        propensity_model = XGBClassifier(**best_propensity_params, 
                                        random_state=random_state,
                                        use_label_encoder=False, 
                                        eval_metric='logloss')
        outcome_model = XGBRegressor(**best_outcome_params, 
                                    random_state=random_state)
        
        # Initialize DRLearner with cv=3 (default)
        dr_learner = DRLearner(model_regression=outcome_model, 
                              model_propensity=propensity_model,
                              cv=3)
        
        # Fit on training blocks
        dr_learner.fit(y_train, T_train, X=X_train, W=None)
        
        # Step 3: Predict ITE on test block
        print("\nStep 4: Estimating ITE on test block...")
        ite_test = dr_learner.effect(X_test)
        
        # Store results
        all_ite.append(ite_test)
        all_X.append(X_test)
        all_y.append(y_test)
        all_T.append(T_test)
        
        print(f"Mean ITE on test block: {ite_test.mean() * 100:.4f}%")
    
    # Concatenate all results
    all_ite = np.concatenate(all_ite)
    all_X = np.vstack(all_X)
    all_y = np.concatenate(all_y)
    all_T = np.concatenate(all_T)
    
    return all_ite, all_X, all_y, all_T


def final_model_for_interpretation(X, y, T, final_columns, random_state=SEED):
    """
    Train final model on all data ONLY for interpretation purposes.
    
    NOTE: This model is NOT used for ITE estimation (which comes from rotation).
    It is only used to generate interpretable decision trees showing which 
    features drive treatment effect heterogeneity.
    
    Parameters:
    -----------
    X: numpy array, All features
    y: numpy array, All outcomes
    T: numpy array, All treatment indicators
    final_columns: pandas Index, Column names for features
    random_state: int, Random seed
    
    Returns:
    --------
    model: DRLearner, Fitted model (for interpretation only)
    """
    print(f"\n{'='*60}")
    print("Training final model for interpretation ONLY")
    print("(NOTE: ITE estimates come from rotation, not this model)")
    print(f"{'='*60}")
    
    # Hyperparameter tuning on all data
    print("\nTuning propensity score model on all data...")
    best_propensity_params = bootstrap_hyperparameter_tuning(
        X, y, T, 
        model_type='propensity', 
        n_bootstrap=100,
        random_state=random_state
    )
    
    print("\nTuning outcome model on all data...")
    best_outcome_params = bootstrap_hyperparameter_tuning(
        X, y, T,
        model_type='outcome',
        n_bootstrap=100,
        random_state=random_state
    )
    
    # Train final model
    print("\nTraining final DRLearner model (cv=3)...")
    propensity_model = XGBClassifier(**best_propensity_params, 
                                    random_state=random_state,
                                    use_label_encoder=False, 
                                    eval_metric='logloss')
    outcome_model = XGBRegressor(**best_outcome_params, 
                                random_state=random_state)
    
    final_model = DRLearner(model_regression=outcome_model, 
                           model_propensity=propensity_model,
                           cv=3)
    
    final_model.fit(y, T, X=X, W=None)
    
    print("\nFinal model trained successfully (for interpretation only).")
    print("Use rotation-based ITE estimates for actual treatment effect analysis.")
    
    return final_model


def visualize_cate_tree(model, X, final_columns):
    """
    Visualize CATE using interpretable decision tree.
    
    Parameters:
    -----------
    model: DRLearner, Fitted model
    X: numpy array, Features
    final_columns: pandas Index, Column names
    """
    print("\nGenerating interpretable CATE tree...")
    
    intrp = SingleTreeCateInterpreter(include_model_uncertainty=True, 
                                     max_depth=2, 
                                     min_samples_leaf=500)
    intrp.interpret(model, X=X)
    
    plt.figure(figsize=(25, 5))
    intrp.plot(feature_names=final_columns, fontsize=12)
    plt.tight_layout()
    plt.show()


def save_results(X, y, T, ite, final_columns, outcome_name):
    """
    Save ITE results to CSV file.
    
    Parameters:
    -----------
    X: numpy array, Features
    y: numpy array, Outcomes
    T: numpy array, Treatment indicators
    ite: numpy array, Individual treatment effects
    final_columns: pandas Index, Column names
    outcome_name: str, Name of outcome
    """
    # Create DataFrame
    df_results = pd.DataFrame(X, columns=final_columns)
    df_results['outcome'] = y
    df_results['treatment'] = T
    df_results['CATE'] = ite
    
    # Save to CSV
    filename = f'CATE_results_for_{outcome_name}.csv'
    df_results.to_csv(filename, index=False)
    print(f"\nResults saved to: {filename}")


def main(train_final_model_for_interpretation=True):
    """
    Main pipeline execution.
    
    Parameters:
    -----------
    train_final_model_for_interpretation: bool, default=True
        If True, trains a final model on all data for interpretability visualization.
        If False, skips final model training (saves ~50% time per outcome).
        Note: ITE estimates always come from rotation, regardless of this setting.
    """
    
    print("="*80)
    print("CATE ESTIMATION PIPELINE WITH BOOTSTRAP HYPERPARAMETER TUNING")
    print("="*80)
    
    if not train_final_model_for_interpretation:
        print("\nNOTE: Skipping final model training for interpretation.")
        print("      Only rotation-based ITE estimates will be computed.")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv('FINAL_DATA_T1D_03252025.csv')
    
    # Define baseline covariates
    baseline_covariances = [
        'SEX', 'AGE', 'BMI', 'Systolic blood pressure', 'Diastolic blood pressure',
        'INSURANCE_COVERAGE', 'SMOKING', 'Race_Ethnicity', 'Cancer',
        'Acute/Chronic Pancreatitis', 'Adenocarcinoma of the esophagus', 'Anorexia',
        'atrial fibrillation ', 'breast cancers', 'Chronic kidney disease',
        'colorectal cancers', 'Coronary Artery Disease', 'Cushing Syndrome',
        'Depression ', 'Dyspepsia', 'Feeding Difficulties', 'Gallbladder cancer',
        'Gallbladder Disease', 'GERD', 'Heart Failure ', 'Hyperlipidemia',
        'Hypertension', 'Inflammatory Bowel Disease', 'Kidney cancer',
        'Liver cancer', 'Meningioma', 'Metabolic Syndrome', 'Multiple myeloma',
        'NAFLD/NASH', 'Ovaries cancer', 'pancreatic cancer', 'Prader-Willi Syndrome',
        'prediabetes', 'Pulmonary Embolism ', 
        'Rheumatoid Arthritis/Osteoarthritis (RA/OA)', 'Sleep apnea', 'Stroke/TIA',
        'Thyroid cancer', 'Type 2 Diabetes (T2D)', 'Upper stomach cancer',
        'Uterus cancer', 'Venous Thromboembolism', 'Vitamin D deficiency', 'AD',
        'ADRD', 'Acutekidneyinjury(AKI)', 'alcohol_use_disorder', 'anemia',
        'anxiety', 'asthma', 'b12_deficiency', 'benign_prostatic_hyperplasia',
        'biliary_disease', 'bipolar', 'cardiovascular_disease', 'cataracts', 'COPD',
        'endometrial_cancer', 'exercise', 'gastroparesis', 'glaucoma', 'gout',
        'hearing_impairment', 'hhf', 'hip_pelvic_fracture', 'lower_extremity_ulcers',
        'lung_cancer', 'Bladder cancer', 'MCI', 'Myocardial infarction',
        'Congestive heart failure ', 'Peripheral vascular disease ',
        'Cerebrovascular disease ', 'Dementia ', 'Chronic pulmonary disease ',
        'Rheumatic disease ', 'Peptic ulcer disease ', 'Mild liver disease ',
        'Diabetes without chronic complication ',
        'Diabetes with chronic complication ', 'Hemiplegia or paraplegia ',
        'Renal disease ',
        'Any malignancy, including lymphoma and leukemia, except malignant neoplasm of skin ',
        'Moderate or severe liver disease ', 'Metastatic solid tumor ', 'AIDS/HIV ',
        'Neuropathy', 'obstruction', 'OCD', 'osteoporosis', 'Parkinson',
        'periodontitis', 'prostate_cancer', 'ptsd', 'schizophrenia', 'seizures',
        'sleep_disorder', 'substance_use', 'suicide', 'tbi', 'thyroid_disease', 'VD',
        'vision_impairment', 'Diabetes', 'esrd', 'bariatic_procedure',
        'bariatic_surgery', 'Pregnant', 'at_least_one_weight_related_cormobidities',
        'CCI Score', 'HbA1c', 'HDL', 'LDL', 'Triglycerides', 'Total cholesterol',
        'eGFR', 'insulin', 'metformin', 'TZD', 'DPP4i', 'SGLT2i', 'SUF', 'AGI',
        'MEG', 'ACEIs', 'beta-blocker', 'CCB', 'diuretic',
        'angiotensin-receptor blocker', 'statin', 'lipid-lowering non-statin',
        'NSAIDS', 'Warfarin', 'Direct oral anticoagulant', 'Aspirin',
        'Non-aspirin antiplatelet agents', 'proton pump inhibitor', 'antidepressant',
        'antipsychotics', 'antiparkinson agent', 'benzodiazepines', 'HRT',
        'anti-dementia medication', 'oral steroids', 'opioid', 'TNF inhibitor',
        'immunosuppresants', 'Orlistat', 'Phentermine_topiramate',
        'Bupropion_Naltrexone', 'Lorcaserin'
    ]
    
    T = df['treatment indicator']
    X = df[baseline_covariances]
    
    # Preprocessing
    print("\nPreprocessing data...")
    
    # Dummy encoding for categorical variables
    categorical_cols = ['SMOKING', 'SEX', 'Race_Ethnicity', 'INSURANCE_COVERAGE']
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    
    # Handle missing values in numeric columns
    numeric_cols = ['Systolic blood pressure', 'Diastolic blood pressure', 'BMI',
                   'HbA1c', 'Total cholesterol', 'HDL', 'LDL', 'Triglycerides', 'eGFR']
    
    for col in numeric_cols:
        mean_value = X[col].mean()
        X[col].fillna(mean_value, inplace=True)
    
    # Save column names
    final_columns = X.columns
    
    # Convert to numpy arrays
    X = X.values
    T = T.values
    
    print(f"Data shape: {X.shape}")
    print(f"Treatment distribution: {np.bincount(T)}")
    
    # Process each outcome
    outcome_list = [
        "Hospitalization or ED visit",
        "Diabetic Ketoacidosis",
        "Severe Hypoglycemia",
        "Hyperglycemic Hyperosmolar State"
    ]
    
    for outcome in outcome_list:
        print(f"\n{'='*80}")
        print(f"PROCESSING OUTCOME: {outcome}")
        print(f"{'='*80}")
        
        # Get outcome
        y = df[outcome + '_indicator'].values
        
        print(f"Outcome distribution: {np.bincount(y)}")
        
        # Split data into 3 blocks
        blocks = stratified_3fold_split(X, y, T, random_state=SEED)
        
        # Estimate CATE with rotation
        all_ite, all_X, all_y, all_T = estimate_cate_with_rotation(
            blocks, final_columns, random_state=SEED
        )
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS FOR {outcome} (ROTATION-BASED)")
        print(f"{'='*60}")
        
        # Calculate ATE from rotation-based ITE estimates
        rotation_ate = all_ite.mean()
        rotation_se = all_ite.std() / np.sqrt(len(all_ite))
        rotation_ci_lower = rotation_ate - 1.96 * rotation_se
        rotation_ci_upper = rotation_ate + 1.96 * rotation_se
        
        print(f"\n** PRIMARY RESULTS (from rotation-based ITE) **")
        print(f"  ATE Estimate: {rotation_ate * 100:.4f}%")
        print(f"  95% CI: ({rotation_ci_lower * 100:.4f}%, {rotation_ci_upper * 100:.4f}%)")
        print(f"  Standard Error: {rotation_se * 100:.4f}%")
        print(f"\n** ITE Distribution **")
        print(f"  Mean ITE: {all_ite.mean() * 100:.4f}%")
        print(f"  Std ITE: {all_ite.std() * 100:.4f}%")
        print(f"  Median ITE: {np.median(all_ite) * 100:.4f}%")
        print(f"  Min ITE: {all_ite.min() * 100:.4f}%")
        print(f"  Max ITE: {all_ite.max() * 100:.4f}%")
        print(f"  25th percentile: {np.percentile(all_ite, 25) * 100:.4f}%")
        print(f"  75th percentile: {np.percentile(all_ite, 75) * 100:.4f}%")
        
        # Optionally train final model for interpretation
        if train_final_model_for_interpretation:
            final_model = final_model_for_interpretation(X, y, T, final_columns, random_state=SEED)
            visualize_cate_tree(final_model, X, final_columns)
        else:
            print("\nSkipping final model training and visualization (as requested).")
        
        # Save results (using rotation-based ITE estimates)
        save_results(all_X, all_y, all_T, all_ite, final_columns, outcome)
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")


if __name__ == "__main__":
    # Configuration
    TRAIN_FINAL_MODEL = True  # Set to False to skip interpretability visualization (saves ~50% time)
    
    main(train_final_model_for_interpretation=TRAIN_FINAL_MODEL)
