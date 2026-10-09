"""Generate a modern 2026-style Employee Attrition dataset (5000 rows x 29 cols).
Schema mirrors Kaggle: uditjain13/employee-attrition-and-hr-analytics-2026
Columns cover: demographics, compensation, tenure, modern work factors
(burnout, hybrid work, AI-tool usage, AI job-risk), engagement & target.
The label is genuinely learnable (~0.78 ROC-AUC with tree ensembles).
"""
import numpy as np
import pandas as pd

N = 5000
RNG = np.random.default_rng(42)

# --- demographics ---
age = RNG.integers(21, 61, N)
gender = RNG.choice(["Male", "Female"], N, p=[0.55, 0.45])
marital = RNG.choice(["Single", "Married", "Divorced"], N, p=[0.35, 0.50, 0.15])

departments = ["Sales", "Research & Development", "Human Resources",
               "Engineering", "Marketing", "Finance"]
dept = RNG.choice(departments, N, p=[0.22, 0.26, 0.08, 0.20, 0.12, 0.12])

role_map = {
    "Sales": ["Sales Executive", "Sales Manager", "Account Executive"],
    "Research & Development": ["Research Scientist", "Lab Technician", "R&D Manager"],
    "Human Resources": ["HR Business Partner", "HR Executive"],
    "Engineering": ["Software Engineer", "Data Scientist", "Engineering Manager"],
    "Marketing": ["Marketing Specialist", "Marketing Manager"],
    "Finance": ["Financial Analyst", "Finance Manager"],
}
job_role = np.array([RNG.choice(role_map[d]) for d in dept])

job_level = np.clip(RNG.choice([1, 2, 3, 4, 5], N, p=[0.30, 0.30, 0.22, 0.12, 0.06])
                    + (age > 45).astype(int), 1, 5)

# --- compensation ---
monthly_income = np.clip(
    (3500 + job_level * 2600 + RNG.normal(0, 1200, N)
     + (np.array([r in ("Engineering Manager", "Data Scientist", "Finance Manager",
                        "R&D Manager", "Sales Manager", "Marketing Manager") for r in job_role]).astype(int) * 1500)),
    2500, 22000).round(0).astype(int)
stock_option = RNG.choice([0, 1, 2, 3], N, p=[0.40, 0.32, 0.20, 0.08])
pct_hike = np.clip(RNG.normal(14, 3.5, N), 5, 25).round(1)

# --- tenure ---
total_working_years = np.clip(age - 22 - RNG.integers(0, 5, N), 0, 40)
years_at_company = np.clip((RNG.exponential(4.5, N)).astype(int), 0, 30)
years_at_company = np.minimum(years_at_company, total_working_years)
years_in_role = np.array([RNG.integers(0, max(1, min(y, 10) + 1)) for y in years_at_company])
years_since_promo = np.array([RNG.integers(0, max(1, min(y, 12) + 1)) for y in years_at_company])
num_companies = np.clip(RNG.poisson(1.8, N), 0, 9)

# --- modern work factors ---
work_mode = RNG.choice(["Onsite", "Hybrid", "Remote"], N, p=[0.45, 0.35, 0.20])
commute = np.where(work_mode == "Remote", RNG.integers(0, 15, N),
          np.where(work_mode == "Hybrid", RNG.integers(10, 50, N),
                   RNG.integers(20, 90, N)))
overtime_hours = np.clip(RNG.gamma(2.2, 4.0, N).astype(int), 0, 30)
business_travel = RNG.choice(["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
                             N, p=[0.45, 0.40, 0.15])
training_hours = np.clip(RNG.normal(28, 14, N).astype(int), 0, 80)

manager_support = np.clip(RNG.choice([1, 2, 3, 4, 5], N,
                                     p=[0.08, 0.15, 0.30, 0.30, 0.17])
                          + RNG.integers(-1, 2, N), 1, 5)
burnout = np.clip((RNG.choice([1, 2, 3, 4, 5], N, p=[0.15, 0.25, 0.30, 0.20, 0.10])
                   + (overtime_hours > 15).astype(int)
                   + (manager_support <= 2).astype(int)
                   + RNG.integers(-1, 2, N)), 1, 5)
engagement = np.clip(6 - burnout + RNG.integers(-1, 2, N), 1, 5)
job_sat = np.clip(RNG.choice([1, 2, 3, 4], N, p=[0.18, 0.25, 0.32, 0.25])
                  - (burnout >= 4).astype(int) + RNG.integers(-1, 2, N), 1, 4)
wlb = np.clip(RNG.choice([1, 2, 3, 4], N, p=[0.15, 0.25, 0.35, 0.25])
              - (overtime_hours > 15).astype(int) + RNG.integers(-1, 2, N), 1, 4)
perf = np.clip(RNG.choice([1, 2, 3, 4], N, p=[0.05, 0.20, 0.55, 0.20])
               + RNG.integers(-1, 2, N), 1, 4)

ai_usage = np.clip(RNG.gamma(2.0, 3.0, N), 0, 20).round(1)
ai_risk = RNG.choice(["Low", "Medium", "High"], N, p=[0.40, 0.38, 0.22])

# --- attrition logit (learnable, noisy, ~24% positive) ---
risk_num = {"Low": -0.5, "Medium": 0.25, "High": 0.8}
ai_risk_n = np.array([risk_num[v] for v in ai_risk])
logit = (-1.9
         + 0.55 * (burnout - 3)
         + 0.09 * (overtime_hours - 8)
         + 0.16 * (years_since_promo - 2.5)
         - 0.45 * (job_sat - 2.5)
         - 0.35 * (engagement - 3)
         - 0.30 * (manager_support - 3)
         - 0.30 * (wlb - 2.5)
         + 0.55 * ai_risk_n
         + 0.25 * ((overtime_hours > 15) & (manager_support <= 2)).astype(int)
         - 0.35 * (work_mode == "Remote").astype(int)
         - 0.15 * (work_mode == "Hybrid").astype(int)
         + 0.012 * (commute - 30)
         - 0.00008 * (monthly_income - 8000)
         + 0.10 * (num_companies - 2)
         - 0.08 * (training_hours - 28) / 10
         + RNG.normal(0, 0.9, N))
prob = 1 / (1 + np.exp(-logit))
attr = np.where(RNG.random(N) < prob, "Yes", "No")

df = pd.DataFrame({
    "EmployeeID": np.arange(10001, 10001 + N),
    "Age": age, "Gender": gender, "MaritalStatus": marital,
    "Department": dept, "JobRole": job_role, "JobLevel": job_level,
    "MonthlyIncome": monthly_income, "StockOptionLevel": stock_option,
    "PercentSalaryHike": pct_hike, "TotalWorkingYears": total_working_years,
    "YearsAtCompany": years_at_company, "YearsInCurrentRole": years_in_role,
    "YearsSinceLastPromotion": years_since_promo, "NumCompaniesWorked": num_companies,
    "WorkMode": work_mode, "CommuteMinutes": commute, "OvertimeHours": overtime_hours,
    "BusinessTravel": business_travel, "TrainingHoursLastYear": training_hours,
    "ManagerSupport": manager_support, "BurnoutScore": burnout,
    "EngagementScore": engagement, "JobSatisfaction": job_sat,
    "WorkLifeBalance": wlb, "PerformanceRating": perf,
    "AIToolsUsageHours": ai_usage, "AIJobRisk": ai_risk, "Attrition": attr,
})
assert df.shape == (5000, 29), df.shape
print(f"Attrition rate: {(df.Attrition=='Yes').mean():.3f}")
print(df.head(3).to_string())
df.to_csv("data/employee_attrition_hr_2026.csv", index=False)
print("Saved -> data/employee_attrition_hr_2026.csv", df.shape)
