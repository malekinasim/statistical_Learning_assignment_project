# Math Grade Analysis in Dalarna Municipalities (Sweden)

This project investigates the factors influencing students' performance in mathematics across municipalities in the Dalarna region of Sweden. The analysis is based on various socioeconomic, demographic, educational, and infrastructural variables.

---

## 🎯 Project Objectives

- Analyze the relationship between student performance and regional characteristics.
- Identify and remove highly collinear features using correlation and VIF.
- Fit multiple regression models: OLS, Fixed Effects, and Random Effects.

---

## 📁 Folder Overview

| Folder | Description |
|--------|-------------|
| `data/raw/` | Raw input files (e.g., from Statistics Sweden or PDF metadata) |
| `data/processed/` | Merged, cleaned, and preprocessed data |
| `data/metadata/` | Additional information such as metadata extracted from PDFs |
| `notebooks/` | Jupyter notebooks for initial exploration and modeling |
| `src/` | Python modules for data loading, visualization, and modeling |
| `output/figures/` | Plots and visual outputs |
| `output/models/` | Model summaries and output files |

---

## ⚙️ How to Run

```bash
pip install -r requirements.txt
python run_analysis.py
