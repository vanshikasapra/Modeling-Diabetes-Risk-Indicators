**Modeling Diabetes Risk Indicators**

**Project Overview**
- Applied analytics project focused on modeling and understanding **diabetes risk indicators**
- Uses multiple approaches to analyze the problem from different angles:
  - Logistic Regression (classification)
  - Linear Regression (baseline + regularization comparison)
  - Clustering (unsupervised segmentation)
- Implemented in Python using Jupyter Notebooks
- Final results and business interpretation summarized in a presentation deck

**Objective**
- Identify and model the key health, demographic, and lifestyle indicators associated with diabetes risk
- Compare supervised and unsupervised approaches to generate both predictive and segment-level insights
- Support health-focused decision-making by balancing predictive performance with interpretability

**Project Components**
- `661-Logistic_final.ipynb` — diabetes classification modeling (logistic regression, ROC/AUC, threshold analysis, SMOTE)
- `661-Linear.ipynb` — regression modeling and regularization experiments (Ridge, Lasso, Elastic Net)
- `661-Clustered.ipynb` — clustering and PCA-based segmentation analysis
- `MGSC 661 Final Presentation.pptx` — final presentation with results and recommendations

**Dataset**
- Source file used in notebooks: `tesdata.xlsx`
- Health indicator dataset with variables related to:
  - Demographics (e.g., age, sex, income, education, race)
  - Lifestyle behaviors (e.g., smoking, drinking, exercise)
  - Access to care (e.g., healthcare provider, affordability, checkups)
  - Medical history / conditions (e.g., heart disease, stroke, kidney disease)
  - Physical health indicators (e.g., BMI, mobility-related variables)
- Target variable used in modeling:
  - `diabetes` / `Diabetes` (binary diabetes outcome for classification workflows)

**End-to-End Workflow**

**1) Data Cleaning & Feature Engineering**
- Imported source data from Excel (`tesdata.xlsx`)
- Mapped coded survey responses into analysis-friendly values
- Handled invalid / non-response codes
- Imputed missing values (mean/mode depending on variable type)
- Created dummy variables for categorical features
- Renamed columns for readability and modeling clarity

**2) Logistic Regression (Classification)**
- Built a baseline logistic regression model for diabetes prediction
- Evaluated model using:
  - Confusion matrix
  - ROC curve
  - AUC
- Tested multiple feature interaction terms (examples include):
  - BMI × Age
  - BMI × Sex
  - Smoker × Alcohol Consumption
  - Exercise × BMI
- Removed variables with high p-values to improve model parsimony
- Compared prediction thresholds (0.5 vs 0.3) to reflect healthcare decision priorities
- Applied SMOTE oversampling to address class imbalance and improve minority-class detection

**3) Linear Regression (Baseline + Regularization)**
- Built a linear regression baseline using the same cleaned feature space
- Checked for multicollinearity (including VIF-based diagnostics)
- Evaluated model performance using regression metrics (e.g., R², MAE, MSE)
- Implemented regularization techniques:
  - Ridge
  - Lasso
  - Elastic Net
- Compared coefficients and model behavior across regularized models

**4) Clustering (Unsupervised Segmentation)**
- Selected a subset of health and behavior indicators for clustering
- Standardized features before clustering
- Applied KMeans clustering
- Used silhouette score across multiple values of K to assess cluster quality
- Performed PCA (2D) for cluster visualization
- Interpreted clusters to identify distinct health-risk / behavior patterns across population segments

**5) Model Comparison & Interpretation**
- Compared supervised model outputs with unsupervised segmentation insights
- Used threshold analysis to align classification decisions with healthcare context
- Prioritized minimizing false negatives where appropriate (missing at-risk individuals)

**Key Results / Highlights**
- Logistic regression achieved strong discrimination performance with AUC ≈ 0.804 
- Threshold comparison (0.5 vs 0.3) was used to evaluate the trade-off between sensitivity and false positives
- Clustering + PCA added interpretable population segments beyond prediction alone
- Regularization analysis (Ridge/Lasso/Elastic Net) helped assess feature stability and predictive ceiling

**Tech Stack**
- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- statsmodels
- imbalanced-learn (SMOTE)
- ISLP (for model summaries / utilities used in notebooks)

**Repository Structure**
- `notebooks/661-Logistic_final.ipynb` — logistic regression analysis (classification, ROC/AUC, threshold tuning, SMOTE)
- `notebooks/661-Linear.ipynb` — linear regression baseline + regularization (Ridge/Lasso/Elastic Net)
- `notebooks/661-Clustered.ipynb` — clustering analysis with KMeans, silhouette scoring, and PCA visualization
- `data/tesdata.xlsx` — dataset used by the notebooks
- `presentations/MGSC 661 Final Presentation.pptx` — final presentation deck
- `README.md`
- `.gitignore`

**How to Run**
1. Clone the repository
2. Place `tesdata.xlsx` inside the `data/` folder (or update notebook file paths accordingly)
3. Install dependencies
4. Open Jupyter Notebook / JupyterLab
5. Run notebooks cell-by-cell

**Install Dependencies**
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `statsmodels`
- `imbalanced-learn`
- `openpyxl`
- `ISLP` (if required by your notebook environment)
