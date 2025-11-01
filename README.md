# Data-Driven Surrogate Model for Deformation of a Soft Bellow Actuator

## Project Overview

This project develops and evaluates machine learning surrogate models to predict the deformation behavior of a soft bellow actuator based on geometric and loading parameters. The models serve as computationally efficient alternatives to Finite Element Analysis (FEA) simulations performed in ANSYS.

**MATLAB Version:** R2025b (25.2.0.2998904)

## Dataset Information

The dataset contains **1,134 unique combinations** of input parameters simulated using FEA in ANSYS.

### Input Features (4 parameters):
1. **Height** - Actuator height dimension
2. **Length** - Actuator length dimension
3. **Pressure** - Applied internal pressure
4. **Thickness** - Wall thickness

### Output Variable:
- **Deformation** - Resulting actuator deformation (target variable)

### Data File:
- `Soft_Actuator_FEA_Dataset.csv` (Update the file path in the script)

## Project Structure

The complete analysis is contained in a single MATLAB script:
- **`SR_actuator_Script3.m`** - Main script with all 18 sections (described below)

## Script Sections and Workflow

### 1. Data Import and Verification
```matlab
data = readtable("path/to/Soft_Actuator_FEA_Dataset.csv");
```
- Imports CSV dataset
- Displays first few rows to verify structure
- **Action Required:** Update the file path to match your local directory

### 2. Descriptive Statistics and Visualization
- Computes statistical measures for all numeric variables:
  - Mean, Median, Standard Deviation
  - Minimum and Maximum values
- Creates histogram visualization of deformation distribution
- Helps understand data range and variability

### 3. Reproducibility Setup
```matlab
rng(42); % Sets random seed
```
- Ensures reproducible results across multiple runs

### 4. Feature and Target Definition
- Separates input features (Height, Length, Pressure, Thickness)
- Isolates target variable (Deformation)
- Prepares data matrices for modeling

### 5. Train-Test Split (80-20)
- **Training set:** 80% of data (~907 samples)
- **Test set:** 20% of data (~227 samples)
- Random permutation ensures unbiased split
- Test set provides independent evaluation

### 6. K-Fold Cross-Validation Setup
- **K = 5 folds** for robust hyperparameter tuning
- Each fold uses different train/validation subsets
- Reduces overfitting risk during model selection

### 7. Model Selection and Hyperparameter Tuning

#### A. Random Forest (RF) Cross-Validation
- **Hyperparameter Grid:** Number of trees [10, 25, 50, 100, 200, 400]
- Method: Bagging with decision trees
- Minimum leaf size: 5
- Evaluates CV RMSE for each configuration
- **Output:** Learning curve plot (CV RMSE vs. Number of Trees)

#### B. Gaussian Process Regression (GPR) Cross-Validation
- **Hyperparameter Grid:** Sigma values [0.25, 0.5, 1, 2, 4]
- Kernel: Squared Exponential (RBF)
- Standardization: Enabled
- Evaluates CV RMSE for each sigma value
- **Output:** Learning curve plot (CV RMSE vs. Sigma)

### 8. Final Model Training and Test Evaluation
- Selects best hyperparameters based on minimum CV RMSE
- Trains final models on entire training set
- Evaluates on held-out test set
- **RF Best Configuration:** Optimal number of trees
- **GPR Configuration:** Sigma = 0.4 (hardcoded after tuning)
- Prints test RMSE for both models

### 9. Performance Metrics Calculation

For both RF and GPR models:
- **RMSE** (Root Mean Squared Error) - Overall prediction accuracy
- **MAE** (Mean Absolute Error) - Average absolute deviation
- **R²** (Coefficient of Determination) - Proportion of variance explained

### 10. Parity Plot (Actual vs. Predicted)
- Visualizes prediction quality
- Ideal predictions lie on the diagonal line
- Includes metrics summary overlay
- Color-coded: Blue (RF), Red (GPR)

### 11. Residual Analysis

#### Histograms of Residuals
- Shows error distribution for RF and GPR
- Ideal: Normal distribution centered at zero
- Identifies systematic bias

#### Residuals vs. Predicted Plots
- Scatter plots for both models
- Ideal: Random scatter around zero
- Detects heteroscedasticity or patterns

### 12. Q-Q Plots for Normality Assessment
- Quantile-Quantile plots for RF and GPR residuals
- Validates normality assumption of errors
- Points should follow diagonal line

### 13. Feature Importance Analysis (Random Forest)
- Ranks input features by importance
- Bar chart visualization
- Identifies which parameters most influence deformation
- Useful for engineering insights

### 14. Partial Dependence Plots

#### Height vs. Deformation
- Shows isolated effect of Height on predictions
- Other features held at average values

#### Pressure vs. Deformation
- Shows isolated effect of Pressure on predictions
- Reveals non-linear relationships

### 15. GPR Uncertainty Quantification
- Error bars represent ±2σ confidence intervals
- Visualizes prediction uncertainty for each test sample
- Compares predicted values with true values
- Unique capability of GPR (not available in RF)

### 16. Single Point Predictions with Confidence Assessment

#### Test Case 1: Interpolative Point
```matlab
x_new1 = [5.25, 55, 1250, 1.05]; % Within training data range
```
- Predicts deformation for typical operational conditions
- Reports RF and GPR predictions with uncertainty

#### Test Case 2: Edge/Extrapolation Point
```matlab
x_new2 = [8.0, 100, 5000, 1.2]; % Near or beyond training bounds
```
- Tests model behavior at extreme conditions
- Higher uncertainty expected

### 17. Confidence-Based Recommendations
- **Uncertainty Threshold:** 0.65 (adjustable)
- **Low Uncertainty (< 0.65):** Model prediction is trustworthy
- **High Uncertainty (≥ 0.65):** Recommend FEA validation
- Automated decision support for engineering applications

## How to Run the Script

### Prerequisites
- MATLAB R2025b or compatible version
- Statistics and Machine Learning Toolbox
- Dataset file: `Soft_Actuator_FEA_Dataset.csv`

### Steps
1. **Update File Path:** Modify Line 2 with your dataset location
   ```matlab
   data = readtable("C:\Your\Path\Soft_Actuator_FEA_Dataset.csv");
   ```
2. **Run Script:** Execute `SR_actuator_Script3.m` in MATLAB
3. **View Results:** Multiple figures will be generated automatically
4. **Check Console:** Numerical results and recommendations printed to command window

## Expected Outputs

### Console Outputs
- Descriptive statistics table
- RF Test RMSE
- GPR Test RMSE
- Interpolative predictions (RF and GPR with uncertainty)
- Edge predictions (RF and GPR with uncertainty)
- Confidence assessments and FEA recommendations

### Visualizations (11 Figures)
1. Histogram of Deformation distribution
2. RF: CV RMSE vs. Number of Trees
3. GPR: CV RMSE vs. Sigma
4. Parity Plot (Actual vs. Predicted) with metrics
5. RF Residuals Histogram
6. GPR Residuals Histogram
7. RF Residuals vs. Predicted scatter
8. GPR Residuals vs. Predicted scatter
9. Q-Q Plots (RF and GPR residuals)
10. Feature Importance bar chart
11. Partial Dependence: Height effect
12. Partial Dependence: Pressure effect
13. GPR Test Predictions with Uncertainty bands

## Interpreting Results

### Model Selection
- Compare RMSE, MAE, and R² values
- Lower RMSE/MAE = Better accuracy
- Higher R² = Better fit (closer to 1.0)
- Consider GPR for uncertainty quantification needs

### Residual Analysis
- **Good Model:** Residuals normally distributed, centered at zero, no patterns
- **Poor Model:** Skewed residuals, systematic bias, heteroscedasticity

### Feature Importance
- Identifies most influential design parameters
- Guides design optimization efforts
- Informs sensitivity analysis

### Uncertainty Assessment
- GPR provides probabilistic predictions
- Use uncertainty threshold to decide: Trust model OR Run FEA
- Critical for safety-critical applications

## Key Advantages of This Approach

✅ **Computational Efficiency:** ML models predict in milliseconds vs. hours for FEA  
✅ **Robust Evaluation:** K-fold CV + independent test set prevent overfitting  
✅ **Comprehensive Diagnostics:** Multiple visualization and statistical checks  
✅ **Uncertainty Quantification:** GPR provides confidence intervals  
✅ **Reproducibility:** Fixed random seed ensures consistent results  
✅ **Interpretability:** Feature importance and partial dependence analysis  
✅ **Decision Support:** Automated confidence assessment for predictions  

## Recommendations for Use

1. **Model Deployment:** Use GPR for predictions requiring uncertainty estimates
2. **Design Space Exploration:** RF faster for batch predictions
3. **Validation:** Always verify edge-case predictions with FEA
4. **Model Updates:** Retrain periodically as more FEA data becomes available
5. **Threshold Tuning:** Adjust uncertainty threshold (0.65) based on application risk tolerance

## Future Enhancements

- Export trained models for deployment (`.mat` files)
- Implement additional algorithms (Support Vector Regression, Neural Networks)
- Multi-objective optimization integration
- Interactive GUI for predictions
- Automated hyperparameter optimization (Bayesian optimization)

## Author

Sudheer Raj


## Acknowledgments

FEA simulations performed using ANSYS. Dataset generated as per project assignment requirements.
