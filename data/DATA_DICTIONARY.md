# DATA DICTIONARY — Employee Attrition HR 2026 (5000 × 29)

Mirrors Kaggle: **uditjain13/employee-attrition-and-hr-analytics-2026**
Link: https://www.kaggle.com/datasets/uditjain13/employee-attrition-and-hr-analytics-2026

| # | Column | Type | Meaning / values |
|---|--------|------|------------------|
| 1 | EmployeeID | int | Unique ID (dropped before training) |
| 2 | Age | int | 21–60 |
| 3 | Gender | cat | Male / Female |
| 4 | MaritalStatus | cat | Single / Married / Divorced |
| 5 | Department | cat | Sales, Research & Development, Human Resources, Engineering, Marketing, Finance |
| 6 | JobRole | cat | Role within department (14 roles) |
| 7 | JobLevel | int | 1–5 seniority |
| 8 | MonthlyIncome | int | $2,500–22,000 |
| 9 | StockOptionLevel | int | 0–3 |
| 10 | PercentSalaryHike | float | 5–25 % |
| 11 | TotalWorkingYears | int | 0–40 |
| 12 | YearsAtCompany | int | 0–30 |
| 13 | YearsInCurrentRole | int | 0–15 |
| 14 | YearsSinceLastPromotion | int | 0–12 (promotion drought) |
| 15 | NumCompaniesWorked | int | 0–9 |
| 16 | WorkMode | cat | **Onsite / Hybrid / Remote** (modern) |
| 17 | CommuteMinutes | int | 0–90 |
| 18 | OvertimeHours | int | extra h/month, 0–30 |
| 19 | BusinessTravel | cat | Non-Travel / Travel_Rarely / Travel_Frequently |
| 20 | TrainingHoursLastYear | int | 0–80 |
| 21 | ManagerSupport | int | 1–5 |
| 22 | BurnoutScore | int | 1–5 (modern) |
| 23 | EngagementScore | int | 1–5 |
| 24 | JobSatisfaction | int | 1–4 |
| 25 | WorkLifeBalance | int | 1–4 |
| 26 | PerformanceRating | int | 1–4 |
| 27 | AIToolsUsageHours | float | AI-tool use h/week (modern) |
| 28 | AIJobRisk | cat | **Low / Medium / High** perceived automation risk (modern) |
| 29 | Attrition | target | Yes (left) / No (stayed) — ~21% Yes |

To use the REAL Kaggle file: download `employee_attrition_hr_2026.csv` from the link
above and replace `data/employee_attrition_hr_2026.csv`, then run `python train.py`.
Column names are aligned, so the app + training work unchanged.
