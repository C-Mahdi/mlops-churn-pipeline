#  Churn Prediction MLOps Project

This project implements a complete end-to-end MLOps pipeline for a churn
prediction model.\
It starts from modular Python code built in a Jupyter Notebook, then
moves into production-ready automation with:

-   Makefile for automated tasks\
-   FastAPI to expose ML endpoints\
-   Streamlit for interactive visualization\
-   Versioning, encoders, scalers, and artifact management

## 📁 Project Structure

    .
    ├── data
    │   └── raw
    │       ├── churn-bigml-20.csv
    │       └── churn-bigml-80.csv
    │
    ├── models
    │   ├── versions/
    │   ├── churn_model.joblib
    │   ├── columns_order.joblib
    │   ├── encoder_area.joblib
    │   ├── encoder_state.joblib
    │   ├── scaler.joblib
    │   └── model_metadata.json
    │
    ├── notebook
    │   └── Churrn.ipynb
    │     
    ├── app.py
    ├── streamlit_app.py
    ├── main.py
    ├── evaluate_with_scores.py
    ├── model_pipeline.py
    ├── version_manager.py
    ├── Makefile
    ├── requirements.txt
    └── .gitignore

##  Features

-   Modular ML pipeline\
-   Automated Makefile workflow\
-   FastAPI model serving\
-   Streamlit UI\
-   Saved ML artifacts

##  Installation

``` bash
git clone https://github.com/C-Mahdi/mlops-churn-pipeline.git
cd mlops-churn-pipeline
```

### 1. Create a virtual environment

``` bash
python -m venv mlops_env
source mlops_env/bin/activate
```

### 2. Install dependencies

``` bash
pip install -r requirements.txt
```

## 🛠 Using the Project

### 1️⃣ Train the model

``` bash
make train
```

### 2️⃣ Run FastAPI

``` bash
uvicorn app:app --reload
```

### 3️⃣ Run Streamlit

``` bash
streamlit run streamlit_app.py
```

##  What I Learned

-   Code modularization\
-   Model versioning\
-   Automation with Makefile\
-   Serving ML with FastAPI\
-   UI visualization with Streamlit\
-   Managing models, encoders, and scalers
