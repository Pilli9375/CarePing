# ⚡ CarePing — Never Lose a Patient in Line
**Datathon 2026 | Track 1: Healthcare Operations (`Health — The Forgotten Patient`)**

* **Team Name:** AAdharsha Kutumbam
* **Team Members:** Naga Adithya V (`23BCE9062`) & Harsha Vardhan Vurtha (`23BCE9129`)
* **Repository:** [https://github.com/Pilli9375/CarePing](https://github.com/Pilli9375/CarePing)

---

## 🎯 Problem Statement
In busy hospitals, non-critical patients get pushed to the back of the queue repeatedly whenever emergency cases arrive—until they feel forgotten and walk out without ever seeing a doctor (**Left Without Being Seen / LWBS**). Furthermore, even after treatment finishes, patients sit in beds for nearly an hour waiting for discharge paperwork, blocking capacity for waiting-room patients.

### Key Evidence Discovered in the Dataset (`15,012 Total Records`)
1. **Waiting Room Starvation:** Low-triage patients wait **71.69 minutes** on average ($4.33\times$ longer than Critical patients at $16.55\text{ min}$), stretching up to **153 minutes** in **Pediatrics (D03)** and **Laboratory (D07)**.
2. **Patient Walkouts (LWBS):** **242 encounters (3.03%)** ended with the patient walking out before care—peaking at **4.77% in Outpatient (D11)**. Alarmingly, **39 walkouts were Critical or High-acuity emergency patients**.
3. **Post-Treatment Boarding Bottleneck:** Across all 8,000 visits, patients spend **52.23 minutes** waiting to be discharged *after* clinical care has already ended—exceeding active treatment time (**47.72 minutes**).
4. **The Urgent Lab Paradox:** Urgent diagnostic tests (**69.49 min** mean turnaround) show zero statistical prioritization over Routine tests (**69.85 min**, $p = 0.8805$). Urgent blood tests actually take **78.23 minutes** vs. **72.27 minutes** for routine blood tests.
5. **Unseen Chronic Registry:** **45 registered patients (2.25%)** had zero hospital visits in 2025, and **26 of them (57.8%)** suffer from chronic illnesses (Diabetes, Hypertension, Cardiac, Asthma).

---

## 💡 Our Solution: CarePing
**CarePing** is a lightweight, real-time queue intelligence and nurse alert system that:
1. **Predicts Walkout Risk Live:** Uses our trained multivariate wait-time estimator to score every waiting patient's walkout risk ($0\text{–}99\%$) and flags low-triage patients hitting the **45-minute threshold** so nurses can perform a quick check-in or fast-track escalation.
2. **Eliminates Discharge Boarding:** Routes medically cleared patients to a dedicated **Discharge Lounge**, cutting post-service bed boarding by 50% and reclaiming **~3,480 bed-hours annually** without buying new beds or hiring extra staff.
3. **Dedicated STAT Diagnostic Lane:** Separates Urgent blood, CT, and ECG tests from routine batch queues with a 30-minute SLA alert.

---

## 📂 Repository Structure
```text
CarePing/
├── index.html                           # Main CarePing Interactive Web Application (GitHub Pages ready)
├── forgotten_patient_dashboard.html     # Standalone Offline Dashboard Copy
├── train_and_evaluate_model.py          # ML Model Training (80/20 Split), ANOVA & Weight Export Script
├── forgotten_patient_analysis.py        # Forensic Data Quality Audit, EDA & High-Res Chart Generator
├── FACULTY_VIVA_QA_GUIDE.md             # Step-by-Step Viva & Judge Q&A Defense Guide
├── requirements.txt                     # Python Dependencies
├── data/                                # Official Datathon Relational Dataset (15,012 records)
│   ├── patients.csv                     # 2,000 registered patients
│   ├── departments.csv                  # 12 hospital departments
│   ├── encounters.csv                   # 8,000 hospital visits
│   ├── diagnostic_tests.csv             # 5,000 lab & imaging orders
│   └── README.txt                       # Dataset documentation
├── outputs/                             # Generated Model Weights, Audit CSVs & High-Res Visuals
│   ├── model_metrics_and_weights.json   # Trained OLS weights, R^2, MAE, RMSE & ANOVA table
│   ├── trained_wait_model_coefficients.csv
│   ├── department_operational_scorecard.csv
│   ├── walkout_lwbs_patients.csv        # All 242 walkout encounters
│   ├── unseen_chronic_registry.csv      # All 45 unseen registered patients
│   ├── chart1_triage_dept_wait.png
│   ├── chart2_lwbs_and_los_breakdown.png
│   ├── chart3_diagnostic_tat_paradox.png
│   └── chart4_data_quality_and_variance.png
└── docs/                                # Presentation & Technical Report Deliverables
    ├── DATATHON-Round1-Idea-PPT-CarePing.pptx
    └── The_Forgotten_Patient_Datathon_Report.pdf
```

---

## 🚀 Quick Start: How to Run Locally

### 1. View the Interactive CarePing Web App
No server is needed! Simply double-click **`index.html`** (or `forgotten_patient_dashboard.html`) in any modern browser (Chrome, Edge, Brave).

### 2. Run the ML Training & Evaluation Pipeline
```bash
pip install -r requirements.txt
python train_and_evaluate_model.py
```
This script loads `data/*.csv` (with `keep_default_na=False` to prevent the `"None"` string corruption trap), runs One-Way ANOVA variance decomposition, trains our Multivariate OLS Queue Model on an **80/20 train-test split**, and exports all weights and metrics to `outputs/`.

### 3. Regenerate All Publication Charts
```bash
python forgotten_patient_analysis.py
```

---

## 📊 Machine Learning Model Summary
* **Model Architecture:** Multivariate Ordinary Least Squares (OLS) Queue Estimator & Walkout Risk Scoring Engine built from scratch in Python (`NumPy` / `SciPy`).
* **Train / Test Split:** 80% Training (`6,400 encounters`) / 20% Testing (`1,600 encounters`), `random_seed=42`.
* **Baseline Comparison:** Predicting the hospital mean wait time ($52.04\text{ min}$) yields a Test MAE of **24.47 minutes** ($R^2 = 0.00$).
* **CarePing Model Performance:**
  * **Variance Explained ($R^2$):** **$0.5238$ ($52.38\%$)** on Full Dataset (`0.5256` Train / `0.5161` Test)
  * **Mean Absolute Error (MAE):** **$13.87\text{ minutes}$** (**44.0% error reduction** over baseline)
  * **Key Finding:** Operational queue factors (`triage_level` $\eta^2 = 46.80\%$, `department` $\eta^2 = 5.58\%$) explain over $52\%$ of wait time variance, whereas patient demographics (`age`, `sex`, `insurance_type`, `chronic_condition`) contribute $<0.30\%$ ($\text{noise}, p > 0.30$).
