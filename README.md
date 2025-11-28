## ⚙️ Accident Risk Prediction App

An **interactive Streamlit web app** that predicts the likelihood of road accidents based on **environmental and road parameters**.
It uses trained machine learning models to provide **mean risk predictions** and a **95% confidence interval**, updating **live** as you adjust inputs.

---

### 🚀 Features

* 🎛️ **Interactive Inputs:** Adjust road and weather conditions using sliders and dropdowns.
* ⚡ **Live Predictions:** The app automatically recalculates the accident risk whenever inputs change.
* 📊 **Confidence Intervals:** Shows both lower and upper bounds of predicted risk at 95% confidence.
* 🌈 **Visual Risk Gauge:** Color-coded indicator (Green → Yellow → Red) based on mean predicted risk.
* 🧠 **Smart Feature Engineering:** Includes transformations and derived features for better model accuracy.
* 💾 **Pre-trained Models:** Uses three LightGBM-based regressors (`model.pkl`, `model_lower.pkl`, `model_upper.pkl`) and a quantile transformer for normalization.

---

### 🏗️ Tech Stack

| Component       | Technology                  |
| --------------- | --------------------------- |
| Frontend UI     | Streamlit                   |
| Data Processing | pandas, numpy               |
| Visualization   | matplotlib                  |
| Model Serving   | LightGBM + joblib           |
| Transformation  | sklearn QuantileTransformer |

---

### 📂 Project Structure

```
accident-risk-predictor/
│
├── model.pkl
├── model_lower.pkl
├── model_upper.pkl
├── quantile_transformer.pkl
├── app.py
└── README.md
```

---
### Screenshots


<img width="1920" height="1080" alt="Screenshot 2025-11-28 222052" src="https://github.com/user-attachments/assets/2a2cf958-842f-4594-9ebb-287785f2851c" />

<img width="1920" height="1080" alt="Screenshot 2025-11-28 222002" src="https://github.com/user-attachments/assets/e6b1b79a-08e6-4508-bc8a-9da1f4bd9feb" />


---

### ⚙️ Installation & Setup

#### 1. Clone the repository

```bash
git clone https://github.com/tejas-130704/ML-Accident-Risk-Predication.git
cd accident-risk-predictor
```

#### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # On Windows
source venv/bin/activate     # On Mac/Linux
```

#### 3. Install dependencies

```bash
pip install streamlit pandas numpy matplotlib joblib scikit-learn lightgbm
```

#### 4. Add model files

Place the following files in the project directory:

* `model.pkl` → trained main model
* `model_lower.pkl` → lower bound model
* `model_upper.pkl` → upper bound model
* `quantile_transformer.pkl` → fitted QuantileTransformer

*(These should match the model features used during training.)*

#### 5. Run the Streamlit app

```bash
streamlit run app.py
```

---

### 🧮 Model Details

* **`model.pkl`**: Predicts central accident risk (mean).
* **`model_lower.pkl` / `model_upper.pkl`**: Estimate the lower and upper confidence bounds.
* **`quantile_transformer.pkl`**: Transforms skewed curvature feature into a normal distribution before inference.

---

### 🖥️ Usage

1. Adjust inputs on the **left panel** (e.g., lanes, curvature, lighting, weather).
2. The **right panel** updates instantly showing:

   * Mean predicted accident risk
   * 95% confidence interval
   * Visual risk bar (orange band)
   * Color-coded risk label (Low, Moderate, High)

---

### 🧠 Example Output

**Predicted Risk (Mean):** `0.627`
**Confidence Interval:** `[0.518, 0.734]`
**Risk Level:** 🔴 High Risk

---

### 🧰 Requirements

| Library      | Version (Recommended) |
| ------------ | --------------------- |
| Python       | 3.9+                  |
| Streamlit    | ≥ 1.36                |
| Pandas       | ≥ 2.0                 |
| NumPy        | ≥ 1.24                |
| scikit-learn | ≥ 1.3                 |
| LightGBM     | ≥ 4.0                 |
| joblib       | ≥ 1.3                 |
| matplotlib   | ≥ 3.8                 |

---

### 🧾 License

This project is licensed under the **MIT License** — feel free to modify and use it in your own work.

---

### 💬 Acknowledgements

* LightGBM for high-performance gradient boosting.
* Streamlit for interactive model deployment.
* scikit-learn for feature scaling and preprocessing.

