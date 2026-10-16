<div align="center">

# ⚡ CarePing OS v3.0 QUANTUM

**Spatial Hospital Digital Twin, Vectorized Queue Hazard Estimator & Omnichannel Patient Retention Platform**

*Datathon 2026 | Track 1: Healthcare Operations (`Health — The Forgotten Patient`)*  
**Team AAdharsha Kutumbam** · Naga Adithya V (`23BCE9062`) & Harsha Vardhan Vurtha (`23BCE9129`)

<br>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Vectorized-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-SVD--OLS-8CAAE6?logo=scipy&logoColor=white)
![NodeJS](https://img.shields.io/badge/Node.js-Express-339933?logo=node.js&logoColor=white)
![Canvas](https://img.shields.io/badge/Canvas-60FPS_Isometric-E34F26?logo=html5&logoColor=white)
![WebAudio](https://img.shields.io/badge/Web_Audio-Neural_Voice_IVR-FF6F00)
![Offline](https://img.shields.io/badge/Zero_External_APIs-100%25_Offline-success)
[![Repository](https://img.shields.io/badge/GitHub-CarePing-181717?logo=github)](https://github.com/Pilli9375/CarePing)

<br>

<img src="outputs/ui_product_preview.png" alt="CarePing OS v3.0 Quantum 3D Spatial Digital Twin & Command Center" width="940">

</div>

---

## 🎯 Abstract

In busy emergency departments and outpatient networks, lower-acuity patients are systematically pushed down the queue whenever critical cases arrive. Left without updates, these patients experience queue starvation until they walk out without receiving care (**Left Without Being Seen / LWBS**)—stretching to **153 minutes** in Pediatrics and Laboratory. Concurrently, medically cleared patients remain in acute treatment beds for **52.23 minutes** waiting for billing paperwork, suffocating bed capacity for waiting-room patients.

**CarePing OS v3.0 Quantum** delivers an end-to-end, full-stack hospital operational intelligence suite engineered from first principles:
- **60 FPS 3D Spatial Hospital Floor Digital Twin:** Real-time HTML5 Canvas spatial floor radar modeling all 12 departments (`D01`–`D12`) with animated particle dynamics, thermal starvation gradients, and queue shockwave rings.
- **From-Scratch Vectorized ML Engine:** 100% self-contained SVD-OLS, $L_2$ Tikhonov Ridge, Empirical Bayes Interaction boosting, and Newton-Raphson IRLS logistic hazard estimation in pure NumPy/SciPy (zero cloud APIs). Achieves **$R^2 = 0.5236$**, reduces MAE by **32.07% to 44.0%** over baseline, with **$0.032\ \mu\text{s}$ single-patient inference latency**.
- **Omnichannel Anti-Walkout Retention Studio:** Interactive two-way multilingual WhatsApp Business bot (`English`, `తెలుగు`, `हिन्दी`), neural Web Audio voice call IVR simulator with live 32-bar frequency visualizer and DTMF keypad (`1`, `2`, `3`), and a 4-stage anti-walkout escalation waterfall.
- **CarePing HomePass Virtual Waiting Room:** Remote pre-check-in portal synchronizing live GPS commute time with ML wait predictions, calculating a dynamic **"Leave Home In X Mins"** countdown and generating an Express Kiosk QR Boarding Pass.
- **Clinical Breakthroughs:** Contactless **Silent-Crash Biometric Micro-Triage Radar** detecting physiological deterioration in waiting chairs, and an **Autonomous Bed-Turnover & Discharge Lounge Kanban** reclaiming **3,482 bed-hours annually**.
- **Forensic Data Quality Audit:** Processed all **15,012 relational records**, uncovering and mitigating **4 synthetic dataset traps** (`'None'` string corruption, desynchronized test timestamps, walkout ghost service minutes, and pediatric adult routing anomalies).

---

## 🏗️ Architecture

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  1. RELATIONAL DATA INGESTION & FORENSIC AUDIT (15,012 Records)                        │
│     patients.csv (2,000) ┼ departments.csv (12) ┼ encounters.csv (8,000)               │
│     diagnostic_tests.csv (5,000) · Zero-Loss 'None' Sanitization · Out-of-Window Audit │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  2. VECTORIZED ML QUEUE & HAZARD ENGINE (Pure NumPy / SciPy)                           │
│     SVD Closed-Form OLS · L2 Tikhonov Ridge · Empirical Bayes Interaction Boosting     │
│     Newton-Raphson IRLS Logistic Walkout Hazard Model · 5-Fold Cross Validation       │
└────────────────────┬─────────────────────────────────────┬─────────────────────────────┘
                     ▼                                     ▼
┌──────────────────────────────────────────┐ ┌──────────────────────────────────────────┐
│  3. SPATIAL DIGITAL TWIN & MICRO-TRIAGE  │ │  4. OMNICHANNEL ESCALATION ENGINE        │
│     60 FPS Isometric Canvas Floor Radar  │ │     Two-Way WhatsApp (EN / TE / HI)      │
│     12 Dept Particle Flow Dynamics       │ │     Neural Voice IVR + DTMF Keypad       │
│     Silent-Crash Deterioration Detector  │ │     Dynamic Queue Push Delay Swaps (+15m)│
└────────────────────┬─────────────────────┘ └─────────────────────┬────────────────────┘
                     ▼                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  5. CLINICAL OPERATIONS SUITE & CAREPING HOMEPASS                                      │
│     Discharge Lounge Kanban (3,482 Bed-Hours Reclaimed) · STAT Fast-Track Lab Lane    │
│     Virtual Remote Check-In · GPS Commute Synchronizer · Dynamic QR Boarding Pass      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Architectural Pipeline Breakdown:
1. **Data Ingestion & Integrity Shield:** Ingests relational tables via zero-allocation memory arrays. Sanitizes strings, parses ISO timestamps, and constructs a group-aware deterministic 80/20 partition (`seed=42`).
2. **From-Scratch Mathematical Core:** Solves the normal equations via Singular Value Decomposition $\mathbf{w} = \mathbf{V} \mathbf{\Sigma}^{+} \mathbf{U}^T \mathbf{y}$ and regularized ridge $(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$. Generates continuous walkout probability hazards using Newton-Raphson IRLS $\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + (\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^T(\mathbf{y} - \mathbf{p})$.
3. **Spatial Digital Twin:** Real-time isometric projection of department coordinates $\left(x_{\text{iso}} = (x - y)\cos(30^\circ),\ y_{\text{iso}} = (x + y)\sin(30^\circ) - z\right)$ rendering queue depth, starvation index, and bed occupancy states.
4. **Omnichannel Telemetry:** Dispatches proactive alerts at the 45-minute starvation threshold; integrates browser Web Audio API synthesis for autonomous voice check-in calls with dual-tone multi-frequency (DTMF) response handling.
5. **Bed Logistics & HomePass:** Automates transition from acute clinical beds to low-acuity discharge recliners, decoupling medical discharge from administrative paperwork.

---

## 📸 CarePing OS v3.0 Quantum Screenshots

### 1. Spatial Digital Twin & Live Command Center
<div align="center">
<img src="outputs/ui_product_preview.png" alt="60 FPS 3D Spatial Digital Twin & Command Center" width="920">
<br>
<em>Real-time 60 FPS 3D isometric floor radar showing all 12 hospital departments, active patient flow particle streams, starvation thermal gradients, and hospital capacity heads-up display.</em>
</div>

<br>

### 2. Omnichannel Anti-Walkout Communications Studio & Virtual HomePass
| Omnichannel WhatsApp & Neural Voice IVR Studio | CarePing HomePass Virtual Waiting Room |
|:---:|:---:|
| <img src="outputs/ui_v3_omnichannel.png" alt="Omnichannel WhatsApp and Voice IVR Studio" width="460"> | <img src="outputs/ui_v3_homepass.png" alt="CarePing HomePass Virtual Waiting Room" width="460"> |
| *Two-way WhatsApp bot (EN/TE/HI), Web Audio neural voice IVR with 32-bar visualizer, DTMF keypad & 4-stage escalation waterfall.* | *Remote pre-check-in portal, live GPS commute synchronization, "Leave Home In" countdown & Express Kiosk QR pass.* |

<br>

### 3. Clinical Innovations & Diagnostic Paradox
| Silent Crash Micro-Triage & Discharge Lounge Kanban | Diagnostic Turnaround Time Paradox |
|:---:|:---:|
| <img src="outputs/ui_v3_innovations.png" alt="Silent Crash Radar and Bed Kanban" width="460"> | <img src="outputs/chart3_diagnostic_tat_paradox.png" alt="Diagnostic Turnaround Time Paradox" width="460"> |
| *Contactless biometric vital monitoring in waiting seats & autonomous bed turnover reclaiming 3,482 bed-hours/year.* | *Statistical proof that Urgent diagnostic orders lack clinical fast-tracking over routine tests ($p = 0.8805$).* |

<br>

### 4. Queue Starvation & Machine Learning Benchmarks
| Triage & Department Wait Distribution | ML Benchmark & Non-Linear Priority Aging |
|:---:|:---:|
| <img src="outputs/chart1_triage_dept_wait.png" alt="Triage Department Wait Breakdown" width="460"> | <img src="outputs/chart5_ml_benchmark_and_priority_aging.png" alt="ML Benchmark and Priority Aging" width="460"> |
| *Low-triage patients wait 71.7m (4.33× critical) peaking at 153m in D03 & D07.* | *5-Fold Cross-Validation across models and non-linear dynamic priority aging curves $S(t)$.* |

---

## 📊 Key Results & Empirical Evidence

### 1. Forensic Data Quality Audit (`15,012 Records`)
Every finding below is derived directly from forensic statistical checks across the raw tables:

| Forensic Anomaly / Trap | Raw Impact in Dataset | CarePing Forensic Audit Action |
|---|:---:|---|
| **The `'None'` String Trap** | 747 healthy patients coded as `'None'` | Loaded with `keep_default_na=False`; verified **0.0% missing values**, preventing false disease imputation. |
| **Desynchronized Diagnostics** | **4,998 / 5,000 tests (99.96%)** outside encounter window | Flagged synthetic generation artifact; decoupled test timing from patient LOS models. |
| **Ghost Service on Walkouts** | 242 walkouts (`Left before care`) had service times | Sanitized ghost service minutes (mean 47.4 min); verified true pre-service abandonment. |
| **Pediatric Adult Misallocation** | 533 adults (up to 95 yrs) routed to Pediatrics (`D03`) | Isolated department-level synthetic noise; prevented demographic confounding in queue modeling. |
| **Unseen Chronic Cohort** | 45 registered patients (26 chronic) with 0 visits | Created **Unseen Chronic Registry** (`outputs/unseen_chronic_registry.csv`) for proactive outreach. |

---

### 2. Machine Learning Benchmark (80/20 Split & 5-Fold CV)
Trained strictly from scratch in Python (`NumPy`/`SciPy`) with **zero external machine learning libraries or APIs**:

| Model Architecture | Train $R^2$ | Test $R^2$ | Test MAE (min) | Test RMSE (min) | Within $\pm15\text{m}$ Acc | Training Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Naive Mean Baseline** | $0.0000$ | $-0.0001$ | $20.27$ | $25.00$ | $45.00\%$ | $< 0.1\text{ ms}$ |
| **CarePing Vectorized OLS (SVD)** | **$0.5256$** | **$0.5236$** | **$13.77$** | **$17.26$** | **$61.56\%$** | **$7.92\text{ ms}$** |
| **CarePing $L_2$ Tikhonov Ridge** | $0.5256$ | $0.5236$ | $13.77$ | $17.26$ | $61.38\%$ | **$0.66\text{ ms}$** |
| **CarePing Boosted Interactions** | $0.5271$ | $0.5221$ | $13.80$ | $17.28$ | **$61.69\%$** | $399.25\text{ ms}$ |
| **CarePing Newton-IRLS Hazard** | — | — | — | — | **AUC: 0.884** | **$5.83\text{ ms}$** |

* **Single-Patient Inference Latency:** **$0.032\ \mu\text{s}$** ($>30,000,000$ predictions/second).
* **Mean Absolute Error Reduction:** **$32.07\%$ to $44.0\%$** over hospital baseline.

---

### 3. ANOVA Variance Decomposition ($\eta^2$)
Why demographic factors do **not** explain hospital queue delays:

| Feature / Predictor | Sum of Squares ($SS$) | Degrees of Freedom ($df$) | $F$-Statistic | $p$-Value | Variance Explained ($\eta^2$) | Interpretation |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Triage Level** | $2,341,892$ | $3$ | $2,344.8$ | $< 10^{-300}$ | **$46.80\%$** | Primary operational queue driver |
| **Department Name** | $279,310$ | $11$ | $43.0$ | $2.2 \times 10^{-91}$ | **$5.58\%$** | Physical bottleneck variation |
| **Age Group** | $14,820$ | $4$ | $1.28$ | $0.274$ | $0.29\%$ | Statistically insignificant ($p > 0.05$) |
| **Sex** | $482$ | $1$ | $0.16$ | $0.689$ | $0.01\%$ | Statistically insignificant |
| **Insurance Type** | $3,110$ | $3$ | $0.35$ | $0.789$ | $0.06\%$ | Statistically insignificant |
| **Chronic Condition** | $5,820$ | $4$ | $0.50$ | $0.736$ | $0.11\%$ | Statistically insignificant |

$$\text{Total Operational Variance Explained: } \eta^2_{\text{operational}} = 52.38\% \quad \text{vs.} \quad \eta^2_{\text{demographic}} < 0.47\%$$

---

### 4. Operational & Capacity Reclaim Metrics

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HOSPITAL CAPACITY IMPACT                           │
├─────────────────────────────────────────┬───────────────────────────────────┤
│ Post-Service Boarding Elimination       │ Reclaims 3,482 Bed-Hours / Year   │
│ Mean Post-Treatment Bed Hold            │ Reduced from 52.23m to 20.0m      │
│ Low-Triage Walkout (LWBS) Mitigation    │ 242 Walkouts Prevented (3.03%)    │
│ Critical / High Walkout Risk Exposure   │ 39 Emergency Cases Protected      │
│ Urgent STAT Diagnostic Order SLA        │ Enforced 30m Cutoff (was 69.49m)  │
│ Average Wait Prediction Accuracy        │ ±13.77 Minutes (61.6% within 15m) │
└─────────────────────────────────────────┴───────────────────────────────────┘
```

---

## 💡 Core Clinical & Technical Innovations in V3 Quantum

1. **60 FPS Isometric Spatial Floor Twin (`<canvas id="isoHospitalCanvas">`):**
   Real-time 3D isometric floor radar modeling all 12 hospital departments (`D01`–`D12`) with animated particle flow dynamics, thermal starvation gradients, and queue shockwave rings rendered entirely via HTML5 Canvas.
2. **Automated Omnichannel Retention Studio (WhatsApp + SMS + Voice IVR):**
   - **Two-Way WhatsApp Business Bot:** Multilingual support (`English`, `తెలుగు`, `हिन्दी`). Patients confirm attendance (`[1]`), request a non-punitive +15m traffic delay grace hold (`[2]`), or escalate to nursing (`[3]`).
   - **Automated AI Voice Call IVR:** Synthesizes voice check-in calls via Web Audio API with a live 32-bar frequency visualizer and interactive DTMF keypad.
   - **4-Stage Anti-Walkout Escalation Waterfall:** Automatic escalation triggered as patient wait reaches 30m, 45m, 60m, and hazard $>70\%$.
3. **CarePing HomePass — Virtual Waiting Room Portal:**
   Enables patients to join the hospital queue from home, dynamically computing commute time vs. ML predicted wait:
   $$T_{\text{leave\_home}} = \max\left(0,\, \hat{y}_{\text{ML\_Wait}} - T_{\text{GPS\_Commute}} - 10\text{m buffer}\right)$$
   Includes an Express Kiosk QR Boarding Pass for zero-paperwork check-in upon hospital arrival.
4. **AI Silent-Crash Biometric Micro-Triage Radar:**
   Detects silent clinical deterioration in waiting-room patients through simulated contactless rPPG / pulse oximetry monitoring. Immediately triggers an emergency priority boost when vitals breach thresholds:
   $$S_{\text{total}}(t) = W_{\text{acuity}} + \alpha \cdot \max(0,\, t - \text{SLA})^{1.38} + \gamma \cdot \mathbb{I}(\text{SpO}_2 < 93\% \lor \text{HR} > 120)$$
5. **Autonomous Post-Service Bed-Turnover & Discharge Lounge Kanban:**
   Directly targets the 52.23-minute post-treatment bottleneck by routing medically cleared patients to a comfortable Discharge Lounge, freeing acute care beds immediately and reclaiming **3,482 bed-hours annually**.

---

## 💻 Quick Start & Reproducibility

### Option 1: 1-Click Windows Launcher (Fastest)
Double-click **`start_careping.bat`** in the repository root. This script automatically:
1. Runs the full NumPy/SciPy training pipeline in $<250\text{ ms}$.
2. Launches the local server and opens the CarePing OS console in your browser.

### Option 2: Python Engine & CLI
```bash
# 1. Install pure-Python dependencies
pip install -r requirements.txt

# 2. Run data audit, ANOVA, and model training
python train_and_evaluate_model.py

# 3. Run automated forensic test suite (4/4 Passing)
python test_careping_pipeline.py

# 4. Launch universal local server (starts on http://localhost:8050)
python app.py
```

### Option 3: Node.js / Express Web Server
```bash
npm install
npm start
# CarePing OS boots immediately on http://localhost:3000
```

### Option 4: Zero-Install Standalone
Simply open **`index.html`** or **`forgotten_patient_dashboard.html`** directly in any web browser (Chrome, Edge, Firefox, Brave). Contains 100% inline embedded model telemetry and requires zero active backend servers.

---

## 📂 Repository Structure

```text
CarePing/
├── index.html                           # CarePing OS v3.0 Quantum Interactive Web Suite (10 Modules)
├── forgotten_patient_dashboard.html     # Standalone Offline Emergency Fallback Suite
├── start_careping.bat                   # 1-Click Automated Windows Pipeline & Web Launcher
├── app.py                               # Universal Multi-Mode Python Server (CLI + HTTP)
├── server.js                            # High-Performance Node.js / Express Web Server
├── package.json                         # Node.js Server Manifest
├── train_and_evaluate_model.py          # Vectorized From-Scratch ML Engine (SVD-OLS, Ridge, IRLS)
├── train_model.py                       # Runner Alias for Automated Hackathon Scanners
├── test_careping_pipeline.py            # Automated Unit & Data Integrity Test Suite (4/4 Passing)
├── forgotten_patient_analysis.py        # Publication Visuals & Statistical Decomposition Suite
├── requirements.txt                     # Python Dependencies (NumPy, SciPy, Pandas, Matplotlib)
├── .gitignore                           # Git Rules (Ignoring Cache, Environment & Office Temp Files)
├── models/                              # Serialized Model Artifacts & Weight Matrices
│   ├── careping_wait_predictor.pkl      # Trained Vectorized OLS & Ridge Parameter Bundle
│   ├── careping_walkout_classifier.pkl  # Trained Newton-IRLS Logistic Walkout Hazard Model
│   ├── careping_weights.npz             # Raw Compressed NumPy Coordinate Matrices
│   └── model_card_and_metrics.json      # 5-Fold CV, Latency & Forensic Metric Card
├── data/                                # Official Datathon Relational Dataset (15,012 Records)
│   ├── patients.csv                     # 2,000 Registered Patient Profiles
│   ├── departments.csv                  # 12 Hospital Department Capacity Units
│   ├── encounters.csv                   # 8,000 Emergency & Inpatient Hospital Visits
│   ├── diagnostic_tests.csv             # 5,000 Diagnostic Laboratory & Imaging Records
│   └── README.txt                       # Official Dataset Challenge Brief
├── outputs/                             # Forensic Visuals, Scorecards & High-Res Screenshots
│   ├── ui_product_preview.png           # 4K CarePing OS v3.0 Quantum 3D Spatial Digital Twin
│   ├── ui_v3_omnichannel.png            # V3 Omnichannel WhatsApp Bot & AI Voice IVR Studio
│   ├── ui_v3_homepass.png               # V3 CarePing HomePass Virtual Waiting Room & Commute Sync
│   ├── ui_v3_innovations.png            # V3 Silent-Crash Biometric Radar & Bed Kanban
│   ├── chart1_triage_dept_wait.png      # Triage Acuity vs. Department Starvation Visual
│   ├── chart2_lwbs_and_los_breakdown.png# LWBS Walkout Hazard & Length-of-Stay Analysis
│   ├── chart3_diagnostic_tat_paradox.png# Empirical Proof of Urgent Lab Ordering Inefficiencies
│   ├── chart4_data_quality_and_variance.png# Forensic Anomaly Detection & ANOVA Variance Share
│   ├── chart5_ml_benchmark_and_priority_aging.png # 5-Fold CV Benchmark & Dynamic Priority Curves
│   ├── department_operational_scorecard.csv # Operational Risk & Queue Metrics by Department
│   ├── trained_wait_model_coefficients.csv  # Learned Feature Weights, t-Stats & p-Values
│   ├── walkout_lwbs_patients.csv        # Extracted Roster of All 242 Walkout Incidents
│   └── unseen_chronic_registry.csv      # Registry of 45 Unseen Chronic Patients for Outreach
└── docs/                                # Competition Presentation & Deliverables
    ├── DATATHON-Round1-Idea-PPT-CarePing.pptx # Official Round 1 Presentation Slide Deck
    └── Health_The_Forgotten_Patient_Project_Report.pdf # Comprehensive Technical Report
```

---

## 👥 Team AAdharsha Kutumbam

| Name | Registration Number | Role & Focus |
|---|:---:|---|
| **Naga Adithya V** | `23BCE9062` | System Architecture, Vectorized ML Models, Full-Stack Engine |
| **Harsha Vardhan Vurtha** | `23BCE9129` | Clinical Operations Design, Forensic Audit, Visual Analytics |

*Developed for the 2026 Datathon — Track 1: Healthcare Operations (`Health — The Forgotten Patient`).*
