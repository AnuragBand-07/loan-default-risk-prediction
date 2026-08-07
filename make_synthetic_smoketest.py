"""
Generates a small, realistic Lending-Club-SHAPED dataset purely to smoke-test the
notebook end-to-end (verify every cell runs on pandas 3 / sklearn 1.9 / xgboost 3.3).
This is NOT for training the real model -- delete data/accepted_2007_to_2018Q4.csv
afterwards and drop in the real Kaggle file.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 60_000

grades = list("ABCDEFG")
grade_risk = {g: i for i, g in enumerate(grades)}          # A=0 (safe) .. G=6 (risky)
grade = rng.choice(grades, size=N, p=[.16, .29, .27, .16, .07, .035, .015])
sub_grade = np.array([f"{g}{rng.integers(1, 6)}" for g in grade])

# interest rate rises with grade risk (+ noise)
int_rate = 6 + np.array([grade_risk[g] for g in grade]) * 3.2 + rng.normal(0, 1.5, N)
int_rate = np.clip(int_rate, 5, 31)

annual_inc = np.clip(rng.lognormal(11.0, 0.6, N), 8_000, 500_000)
loan_amnt = np.clip(rng.normal(15_000, 8_000, N), 1_000, 40_000).round(-2)
term_months = rng.choice([36, 60], size=N, p=[.72, .28])
installment = (loan_amnt * (int_rate/1200) /
               (1 - (1 + int_rate/1200) ** (-term_months))).round(2)

dti = np.clip(rng.normal(18, 8, N), 0, 45)
fico_low = np.clip(rng.normal(695, 30, N) - np.array([grade_risk[g] for g in grade]) * 6,
                   610, 845).round(0)
fico_high = fico_low + 4
emp_years = rng.choice(
    ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
     "6 years", "7 years", "8 years", "9 years", "10+ years", np.nan],
    size=N)
home = rng.choice(["RENT", "MORTGAGE", "OWN", "ANY"], size=N, p=[.4, .48, .11, .01])
verif = rng.choice(["Verified", "Source Verified", "Not Verified"], size=N)
purpose = rng.choice(
    ["debt_consolidation", "credit_card", "home_improvement", "major_purchase",
     "medical", "small_business", "car", "other"], size=N,
    p=[.55, .2, .07, .04, .03, .03, .03, .05])
states = ["CA", "TX", "NY", "FL", "IL", "NJ", "PA", "OH", "GA", "NC", "VA", "MI"]
addr_state = rng.choice(states, size=N)

delinq_2yrs = rng.poisson(0.3, N)
inq_6m = rng.poisson(0.7, N)
open_acc = np.clip(rng.normal(11, 5, N), 1, 40).round()
total_acc = open_acc + np.clip(rng.normal(13, 7, N), 0, 50).round()
pub_rec = rng.binomial(1, 0.12, N)
revol_bal = np.clip(rng.normal(15_000, 12_000, N), 0, 120_000).round()
revol_util = np.clip(rng.normal(55, 24, N), 0, 130).round(1)
mort_acc = rng.poisson(1.2, N)
pub_rec_bank = (pub_rec & rng.binomial(1, 0.6, N))

# issue / credit-line dates
issue_year = rng.integers(2013, 2018, N)
issue_month = rng.integers(1, 13, N)
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
issue_d = [f"{months[m-1]}-{y}" for m, y in zip(issue_month, issue_year)]
cr_age = rng.integers(2, 30, N)
earliest = [f"{months[rng.integers(0,12)]}-{y-a}" for y, a in zip(issue_year, cr_age)]

# ---- latent default probability (drives a ~0.70 AUC signal) ----
z = (-3.1
     + 0.14 * np.array([grade_risk[g] for g in grade])
     + 0.045 * (int_rate - 12)
     + 0.020 * (dti - 18)
     + 0.9 * (loan_amnt / annual_inc)
     + 0.35 * delinq_2yrs
     + 0.25 * pub_rec
     + 0.006 * (revol_util - 55)
     - 0.004 * (fico_low - 695)
     + rng.normal(0, 1.0, N))            # noise caps the achievable AUC ~0.70
p_default = 1 / (1 + np.exp(-z))
defaulted = rng.binomial(1, p_default)

status = np.where(defaulted == 1, "Charged Off", "Fully Paid").astype(object)
# sprinkle in some unfinished loans that the notebook should DROP
ongoing = rng.random(N) < 0.12
status[ongoing] = rng.choice(["Current", "Late (31-120 days)", "In Grace Period"],
                             size=ongoing.sum())

df = pd.DataFrame({
    "loan_status": status,
    "loan_amnt": loan_amnt,
    "term": [f" {t} months" for t in term_months],
    "int_rate": [f"{r:.2f}%" for r in int_rate],
    "installment": installment,
    "grade": grade,
    "sub_grade": sub_grade,
    "purpose": purpose,
    "initial_list_status": rng.choice(["w", "f"], size=N),
    "application_type": rng.choice(["Individual", "Joint App"], size=N, p=[.95, .05]),
    "issue_d": issue_d,
    "emp_length": emp_years,
    "home_ownership": home,
    "annual_inc": annual_inc.round(0),
    "verification_status": verif,
    "addr_state": addr_state,
    "dti": dti.round(2),
    "delinq_2yrs": delinq_2yrs,
    "earliest_cr_line": earliest,
    "fico_range_low": fico_low,
    "fico_range_high": fico_high,
    "inq_last_6mths": inq_6m,
    "open_acc": open_acc,
    "pub_rec": pub_rec,
    "revol_bal": revol_bal,
    "revol_util": [f"{u:.1f}%" for u in revol_util],
    "total_acc": total_acc,
    "mort_acc": mort_acc,
    "pub_rec_bankruptcies": pub_rec_bank,
    # a couple of extra "leaky" columns to mimic the real 151-col file (should be ignored)
    "recoveries": rng.random(N) * 100,
    "total_pymnt": rng.random(N) * 20000,
})
df.to_csv("data/accepted_2007_to_2018Q4.csv", index=False)
print(f"Wrote synthetic smoke-test data: {df.shape} -> data/accepted_2007_to_2018Q4.csv")
print("Overall default rate:", (df['loan_status'] == 'Charged Off').mean().round(3))
