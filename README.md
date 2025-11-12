# 🌞 Solar Challenge Week 0 — Task 1: Git & Environment Setup

## 🎯 Objective
Get everyone comfortable with **version control** and **environment setup** before working with data.

---

## 🧩 Task Overview

### 1. Initialize Repository
- Create a new GitHub repository named **`solar-challenge-week0`**.
- Clone it locally:
  ```bash
  git clone https://github.com/<your-username>/solar-challenge-week0.git
  cd solar-challenge-week0

# 🌞 Task 2: Data Profiling, Cleaning & Exploratory Data Analysis (EDA)

## 🎯 Objective
The goal of this task is to **profile, clean, and explore** each country's solar dataset (Benin, Sierra Leone, and Togo) to ensure high-quality, consistent data ready for cross-country comparison and regional solar potential ranking.

---

## 🧠 Overview
Each country’s dataset is analyzed and cleaned in its own branch and notebook:
- **Branch name format:** `eda-<country>` (e.g., `eda-benin`)
- **Notebook name format:** `<country>_eda.ipynb`  
- **Cleaned dataset saved as:** `data/<country>_clean.csv`

All cleaned data files are **excluded from version control** by adding `data/` to `.gitignore`.

---

## 🧹 Data Profiling & Cleaning Steps

### 1. **Summary Statistics & Missing-Value Report**
- Generated descriptive statistics using:
  ```python
  df.describe()
  df.isna().sum()


# 🌍 Solar Challenge Week 0 — Task 3: Cross-Country Comparison

## 🎯 Objective
Synthesize the **cleaned solar datasets** from **Benin**, **Sierra Leone**, and **Togo** to identify **relative solar potential** and highlight key differences across countries.

**Branch:** `compare-countries`  
**Notebook:** `compare_countries.ipynb`

---

## 🧠 Overview
This task focuses on integrating the cleaned datasets from multiple countries and performing comparative analysis of solar metrics such as:

- **GHI** (Global Horizontal Irradiance)  
- **DNI** (Direct Normal Irradiance)  
- **DHI** (Diffuse Horizontal Irradiance)

Through **visualizations**, **statistical testing**, and **summary tables**, the goal is to reveal which country shows the strongest potential for solar energy development.

---

## 📂 Input Data
All input files must exist under the `data/` directory:

