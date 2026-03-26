# 🚀 Rawlytics 
**Stop writing boilerplate EDA code. Start extracting insights.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.x-61dafb.svg)](https://reactjs.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.x-38B2AC.svg)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Rawlytics** is a full-stack, intelligent data profiling and machine learning readiness platform. Upload your raw datasets and instantly receive production-grade statistical insights, automated data cleaning recommendations, and feature engineering strategies—all wrapped in a sleek, modern UI.


## ✨ Why Rawlytics?
Data scientists and analysts spend up to 80% of their time cleaning and profiling data. Rawlytics automates the tedious "first steps" of a data pipeline. Whether you are checking for statistical normality, dealing with missing values, or preparing categorical features for a machine learning model, Rawlytics handles the heavy lifting in milliseconds.

## 🛠️ Core Features

### 📊 Deep Statistical Engine
* **Advanced Profiling:** Automatically calculates descriptive statistics, excess kurtosis, and skewness.
* **Normality Testing:** Dynamically runs Shapiro-Wilk (for n ≤ 5000) or Kolmogorov-Smirnov tests to evaluate distribution characteristics.
* **Categorical Entropy:** Calculates Shannon entropy to measure the diversity and concentration of categorical variables.
* **Correlation Matrices:** Generates Pearson correlations for numeric features and Cramér's V for categorical features.

### 🧹 Context-Aware Data Remediation
* **Smart Imputation:** Analyzes missingness rates and feature correlations to recommend and apply the mathematically optimal imputation strategy (Mean, Median, Mode, FFill, BFill).
* **Anomaly Detection:** Flags outliers in real-time using user-configurable IQR or Z-score methodologies.
* **Automated Type Inference:** Heuristically scans columns to identify miscast dates, booleans, or IDs, suggesting optimal data types to drastically reduce memory footprint.

### 🤖 Machine Learning Readiness
* **Feature Importance:** Computes mutual information scores to rank feature relevance against your chosen target variable for both classification and regression tasks.
* **Encoding Recommendations:** Suggests the most efficient categorical encoding techniques (One-Hot, Ordinal, Target/Hashing) based on column cardinality.
* **Class Imbalance Detection:** Scans classification targets for skewed distributions and flags minority classes (under 20%), suggesting remediation strategies like SMOTE.

### ⚡ High-Performance Architecture
* **Stateless Session Management:** Secure, UUID-based session store allows concurrent users to upload and process datasets up to 50MB entirely in memory.
* **Comprehensive EDA Reports:** A single `/report` endpoint asynchronously combines all analyses into one unified, easily consumable JSON payload.

---

## 💻 Tech Stack

**Frontend (Client)**
* **Core:** React.js, Vite
* **Styling:** Tailwind CSS
* **Components:** shadcn/ui
* **Icons:** Lucide React

**Backend (Server)**
* **API Framework:** FastAPI
* **Data Processing:** Pandas, NumPy
* **Statistical Analytics:** SciPy
* **Machine Learning Ops:** Scikit-Learn

---

## 🚀 Getting Started

Follow these steps to get Rawlytics running locally on your machine.

### Prerequisites
* Python 3.9+
* Node.js 18+
* Git

### 1. Clone the Repository
```bash
git clone [https://github.com/MazenHisham1/rawlytics.git](https://github.com/MazenHisham1/rawlytics.git)
cd rawlytics
