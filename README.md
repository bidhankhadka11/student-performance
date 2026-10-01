Predicting Student Performance (Math)

Machine learning project that predicts a student's final Math grade (G3, 0–20) from demographic, social, school-related and earlier-grade attributes, using the UCI Student Performance dataset.

 Team

| Member | GitHub | Part |
| Aleida Ramirez | [@AleidaRamirez2026](https://github.com/AleidaRamirez2026) | Part A: EDA & Preprocessing |
| Bidhan Khadka | [@bidhankhadka11](https://github.com/bidhankhadka11) | Part B: Modeling & Evaluation |

 Dataset

Source: [UCI Machine Learning Repository: Student Performance](https://archive.ics.uci.edu/ml/datasets/Student+Performance) (Cortez & Silva, 2008)
File used:`student-mat.csv` (Mathematics course, `;`-separated)
Size: 395 students × 33 attributes, no missing values
Target: `G3`, the final-period grade (0–20)
Features: school, demographics, family background, study habits, social life, and earlier grades `G1` (1st period) and `G2` (2nd period)

 How to run

Run the notebooks in order: Part A first, then Part B.** Part A creates `processed_data.csv`, which Part B loads.

1. Clone the repo:
   ```bash
   git clone https://github.com/bidhankhadka11/student-performance.git
   cd student-performance
   ```
2. Install the libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. Open the notebooks:
   VS Code: open the `student-performance` folder, install the Jupyter extension (by Microsoft), open a notebook, click Select Kernel and choose your Python 3, then click Run All.
   Jupyter: run `jupyter notebook` in the project folder, open a notebook in the browser tab that appears, then click Run → Run All Cells.
4. Run `A_eda_preprocessing.ipynb` first, then `B_modeling_evaluation.ipynb`.

All randomness uses `RANDOM_STATE = 42`, so results are reproducible.

Part A: EDA & Preprocessing

Exploratory analysis

Descriptive statistics (mean, median, std) for age, absences, study time, failures, G1, G2, G3
Histogram of G3, correlation heatmap, box plots (G3 by study time and failures), G2 vs G3 scatter plot

Preprocessing
One-hot encoded all 17 categorical columns with `pd.get_dummies(drop_first=True)`: 33 → 42 columns (41 features + G3)
Ordinal features (parents' education, 1–5 ratings) kept as integers
Scaling (`StandardScaler`) is applied in Part B **after** the train/test split to avoid data leakage; only Linear Regression needs it
Feature importance via correlation with G3
Exports `processed_data.csv` for Part B

Key findings
Earlier grades dominate: G2 (r = 0.90) and G1 (r = 0.80) are by far the strongest predictors of G3. Every other feature has |r| < 0.4.
38 students (9.6%) have G3 = 0. All of them have 0 recorded absences, which suggests they dropped the course or skipped the final exam. They break the G2 → G3 pattern and are the hardest cases to predict.
Past failures is the strongest non-grade predictor (r = −0.36): median G3 falls from 11 to 7 as failures go from 0 to 3.
Mother's education and wanting higher education are mildly positive; age, going out and travel time are mildly negative. Study time has only a weak effect.

Part B: Modeling & Evaluation

Models: Linear Regression and Random Forest, trained on an 80/20 train/test split
Feature sets: with G1/G2, and without G1/G2 (background factors only)
Metrics: Mean Squared Error (MSE) and R²

| Model | Features | MSE | R² |
|---|---|---|---|
| Linear Regression | with G1/G2 | _TBD_ | _TBD_ |
| Linear Regression | without G1/G2 | _TBD_ | _TBD_ |
| Random Forest | with G1/G2 | _TBD_ | _TBD_ |
| Random Forest | without G1/G2 | _TBD_ | _TBD_ |

_Results and discussion will be added once Part B is complete._

Tools:

Python 3 · pandas · NumPy · matplotlib · seaborn · scikit-learn · Jupyter

Reference:

Cortez, P. & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance.* In Proceedings of the 5th Future Business Technology Conference (FUBUTEC 2008), pp. 5–12. UCI Machine Learning Repository.
