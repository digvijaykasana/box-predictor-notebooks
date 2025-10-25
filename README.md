# Box Predictor — Optima Model (Notebooks)

This repository contains Jupyter notebooks for building and evaluating the **Optima model**, which predicts packaging box counts and sizes from product and order details.

The main notebook `optima_model.ipynb` includes all logic for feature engineering, model training, optimization, and artifact export.  
The resulting model (`model.joblib`, `preprocess.joblib`, `schema.json`) is used downstream in the **ADK agent** and **Streamlit app** repositories.

---

## 📁 Folder Structure
```
box-predictor-notebooks/
├── notebooks/
│   ├── optima_model.ipynb       # main notebook with Optima model
│   ├── train_model.ipynb        # placeholder for retraining logic
│   └── evaluate_model.ipynb     # placeholder for evaluation logic
├── requirements.txt
├── .gitignore
└── .env.example
```

---

## 🚀 Quickstart

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Then open `notebooks/optima_model.ipynb` to explore or rerun the workflow.

---

## Training and Exporting Artifacts

Example snippet to save your trained Optima model and preprocessor:

```python
from joblib import dump

dump(model, "artifacts/model.joblib")
dump(preprocessor, "artifacts/preprocess.joblib")

# Save schema for downstream usage
import json
schema = {"features": list(X_train.columns)}
with open("artifacts/schema.json", "w") as f:
    json.dump(schema, f, indent=2)
```

To reload later:
```python
from joblib import load
model = load("artifacts/model.joblib")
preds = model.predict(X_new)
```

---

## Integration

This repository provides artifacts for:
- **box-predictor-adk** → wraps Optima model via Google ADK for LLM-driven predictions.
- **box-predictor-streamlit** → provides an interactive web interface using Streamlit.

---

## ⚙️ Environment Variables
Example `.env` file:
```
DATA_PATH=./data
MODEL_PATH=./artifacts/model.joblib
```

---

## Dependencies

See `requirements.txt`:
```
pandas
numpy
scikit-learn
xgboost
matplotlib
joblib
jupyter
```
Install them with:
```bash
pip install -r requirements.txt
```

---

## 🧹 .gitignore Highlights
- `.venv/`, `.env`, `data/`, `artifacts/`, caches, and Jupyter checkpoints are ignored.

---

## 🌐 Git Setup
```bash
git init
git add .
git commit -m "Initial commit: Box Predictor — Optima Model"
git branch -M main
git remote add origin https://github.com/digvijaykasana/box-predictor-notebooks.git
git push -u origin main
```
