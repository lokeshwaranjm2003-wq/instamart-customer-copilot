# Instamart Customer Intelligence & Support Copilot

## Overview

An end-to-end Python capstone project using synthetic Instamart customer, product, order, and support-ticket data.

The project combines data cleaning, SQLite analytics, customer segmentation, lapse prediction, support ticket classification, policy retrieval, and a simulated refund agent.

**Technologies:** Python, Pandas, NumPy, SQLite, Scikit-learn, Joblib, Git.

## Reproduce

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd instamart-capstone
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Generate data and run the project

```bash
python generate_data.py
python part_a.py
python run_queries.py
python part_a_analysis.py
python validate_part_a.py
python part_b.py
python part_b_lapse.py
python part_b_model.py
python part_c.py
python part_c_classify.py
python part_c_routing.py
python validate_part_c.py
python part_d.py
python part_d_rag.py
python part_d_assistant.py
python part_d_agent.py
python refund_tools.py
python refund_agent.py
python validate_part_d.py
```

The project uses synthetic data and a local mock LLM. No paid API key is required.

## Part A — Data Cleaning and SQL

Cleaned customer, product, and order data and stored the results in SQLite.

Implemented SQL queries covering customer spending, revenue, product pricing, inactive customers, customer ranking, and cumulative revenue.

**Validation results:**
- Customers: 1,200
- Products: 40
- Cleaned orders: 12,309
- Completed orders: 11,110
- Cancelled orders: 1,199
- Part A validation passed.

## Part B — Customer Segmentation and Lapse Prediction

Performed RFM analysis and K-Means clustering to identify four customer segments.

| Segment | Customers |
|---|---:|
| Regular | 436 |
| Loyal | 419 |
| At Risk | 182 |
| Premium | 138 |

Compared K-Means cluster counts from K=2 through K=8.

Applied PCA for dimensionality reduction.

**Lapse prediction:**
- Cohort customers: 1,156
- Lapsed customers: 443
- Lapse rate: 38.32%
- Training rows: 924
- Testing rows: 232

Trained Logistic Regression and Random Forest classifiers using preprocessing pipelines and hyperparameter search.

| Model | Test F1 |
|---|---:|
| Logistic Regression | 0.7236 |
| Random Forest | 0.7059 |

Selected Logistic Regression and saved the trained pipeline as `outputs/lapse_model.joblib`.

Created a retention list of 10 Premium customers.

## Part C — Support Ticket Triage

Built a support ticket classification pipeline using the supplied mock LLM.

Implemented privacy redaction, token estimation, long-ticket handling, JSON validation, retry logic, token budget demonstration, and support routing.

**Results:**
- Total tickets: 164
- Classified tickets: 147
- Emails redacted: 55
- Phone numbers redacted: 91
- Order IDs redacted: 132
- Total redactions: 278
- Few-shot classification accuracy: 100% on classified tickets
- HIGH priority tickets: 45
- NORMAL priority tickets: 119

Fraud tickets are routed to the security team, Premium tickets to the premium desk, and unsupported or invalid classifications to human review.

Part C validation passed.

## Part D — Policy RAG and Refund Agent

Implemented a policy retrieval system using overlapping text chunks, TF-IDF vectorization, and cosine similarity.

Built a policy question-answering assistant using retrieved context and the supplied mock LLM.

Implemented a simulated refund agent supporting:

- Order lookup
- Policy retrieval
- Refund eligibility calculation
- Refund issuance in a local SQLite state database
- Human escalation
- Duplicate refund prevention

Refund decisions are based on the configured project rules, including late-delivery eligibility and human approval for refunds above INR 1,000.

The system does not process real payments.

Part D validation passed.

## Limitations and Next Steps

- All data is synthetic.
- The LLM is simulated and does not represent a production model.
- Clustering labels are interpretations of customer behaviour.
- Lapse model evaluation uses a fixed historical cutoff and one train/test split.
- Retention targeting requires further leakage-free temporal evaluation.
- Policy retrieval can return partially relevant text.
- Refund processing is simulated and requires stronger authorization, policy enforcement, audit logging, and concurrency controls before production use.

Future improvements include temporal model validation, stronger policy-grounding checks, expanded automated tests, monitoring, and deployment through a web API.
