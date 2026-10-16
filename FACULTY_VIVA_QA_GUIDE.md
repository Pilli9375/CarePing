# 🎓 Faculty & Judge Viva Q&A Defense Guide — Team AAdharsha Kutumbam
**Product Name:** CarePing  
**Team Members:** Naga Adithya V (`23BCE9062`) & Harsha Vardhan Vurtha (`23BCE9129`)

Use this cheat sheet when faculty or judges ask **"How did you do this work?"**, **"How did you clean the data?"**, or **"How did you train your model?"**

---

## Q1: "Where did the numbers in your HTML prototype come from? Did you hardcode them or compute them?"
**Your Answer:**
> "Every single number in our CarePing dashboard was computed directly from the 4 official CSV files (`15,012 total records`: 2,000 patients, 12 departments, 8,000 encounters, and 5,000 diagnostic tests) using our Python pipeline (`train_and_evaluate_model.py` and `forgotten_patient_analysis.py`). We joined `patients`, `departments`, `encounters`, and `diagnostic_tests` on `patient_id`, `department_id`, and `encounter_id`, ran statistical tests and regression modeling in Python, and exported the exact aggregated JSON payload into our standalone HTML file so it runs fast and 100% offline during the demo."

---

## Q2: "What data cleaning did you do? Did you find any traps in the dataset?"
**Your Answer (This will impress the judges immediately because the README specifically tests for this!):**
> "Yes! The dataset `README.txt` warned us not to treat synthetic associations as clinical truth, so we ran a forensic audit first and caught **4 major traps**:
> 1. **The `'None'` String Trap:** In `patients.csv`, 747 healthy patients have their chronic condition written as the literal word `'None'`. If you load `patients.csv` with default `pd.read_csv()`, Pandas misinterprets `'None'` as `NaN` (missing data) and tricks teams into imputing fake diseases! We used `keep_default_na=False` and proved the dataset actually has **0.0% missing values**.
> 2. **Desynchronized Test Timestamps:** When we joined `diagnostic_tests.csv` with `encounters.csv`, we checked if `requested_time` fell between `arrival_time` and `discharge_time`. We found that **4,998 out of 5,000 tests (99.96%)** had timestamps outside the patient's visit window! That proved test timestamps were synthetically generated independently.
> 3. **Ghost Service on Walkouts:** All **242 walkout patients (`Left before care`)** still had `service_start_time` and `service_minutes` (averaging 47.4 minutes) recorded in the table.
> 4. **Adults in Pediatrics:** In Pediatrics (`D03`), 81.1% of encounters (533 visits) were adults over age 18 (up to age 95), proving synthetic department assignment didn't filter by age."

---

## Q3: "How did you train the ML model? What is the target, features, split, and metric?"
**Your Answer:**
> "We built our model **100% from scratch in Python** (`train_and_evaluate_model.py`) using `NumPy` and `SciPy`—no external black-box APIs.
> * **Target Variable ($y$):** Individual patient waiting time (`wait_minutes`).
> * **Input Features ($X$):** One-hot encoded `triage_level` (Critical, High, Medium, Low), `department_name` (all 12 units), standardized `age`, and `arrival_hour`.
> * **Train / Test Split:** We used a deterministic **80/20 split** (`6,400 training encounters` and `1,600 testing encounters` with `seed=42`).
> * **Baseline Model:** A naive baseline that predicts the hospital average wait time (`52.04 minutes`) gets a Mean Absolute Error (MAE) of **24.47 minutes** ($R^2 = 0.00$).
> * **Our Trained Model:** Our Multivariate Ordinary Least Squares (OLS) model achieved **$R^2 = 52.38\%$** ($0.5256$ Train / $0.5161$ Test) and reduced the Mean Absolute Error down to **13.87 minutes**—beating the hospital baseline by **44%**!
> * **Saved Weights:** All learned coefficients, $t$-statistics, and $p$-values are saved in `outputs/trained_wait_model_coefficients.csv` and `outputs/model_metrics_and_weights.json`."

---

## Q4: "Why didn't patient age, gender, or insurance affect waiting time?"
**Your Answer:**
> "We ran a One-Way **ANOVA Variance Decomposition ($\eta^2$)** across every column:
> * `triage_level` explains **46.80%** of wait-time variance ($F = 2,344.8, p < 10^{-300}$).
> * `department_name` explains **5.58%** of wait-time variance ($F = 43.0, p = 2.2 \times 10^{-91}$).
> * In contrast, `age`, `sex`, `insurance_type`, `chronic_condition`, and `home_area` each explained **less than 0.30%** of variance with $p > 0.30$ (statistically insignificant).
> This proves that **operational queue design—not demographic bias—is what causes patients to be forgotten**."

---

## Q5: "How does the model output connect to the CarePing product?"
**Your Answer:**
> "When a patient checks in at reception, CarePing uses our trained weights (`Baseline 10.67m + Triage Penalty + Department Bottleneck Penalty`) to estimate their expected wait and calculate a live **Walkout Risk Score (0–99%)**. Once a low-triage patient crosses **45 minutes** (or 75% walkout risk), CarePing flashes a red alert on the nurse's board so they can do a quick comfort check-in or fast-track them before they walk out."
