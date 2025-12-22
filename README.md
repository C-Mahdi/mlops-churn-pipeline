# Churn Prediction MLOps Project

This repository implements a comprehensive end-to-end MLOps pipeline for churn prediction using machine learning. The project demonstrates best practices in modular code development, automation, deployment, and monitoring.

## 📋 Description

This project builds a complete MLOps pipeline starting from data exploration in Jupyter Notebooks to production deployment. It includes:

- **Modular ML Pipeline**: Clean, reusable code components for data processing, model training, and evaluation.
- **Automation with Makefile**: Streamlined workflows for training, testing, and deployment tasks.
- **FastAPI Endpoints**: RESTful API for model inference and predictions.
- **Streamlit Interface**: Interactive web application for model consumption and visualization.
- **Docker Containerization**: Containerized application for easy deployment and scalability.
- **CI/CD with GitHub Actions**: Automated Docker image building and deployment triggered by Dockerfile changes.
- **ELK Stack Monitoring**: Elasticsearch, Logstash, and Kibana for logging and monitoring the application.

The pipeline uses the BigML churn dataset to predict customer churn based on telecom data.

##  Features

- Modular Python code for maintainability and reusability
- Automated training and evaluation workflows
- RESTful API for model serving
- Interactive Streamlit dashboard
- Docker containerization for deployment
- Automated CI/CD pipelines
- Comprehensive logging and monitoring with ELK stack
- Model versioning and artifact management with MLflow
- Data preprocessing with encoders and scalers

## 📁 Project Structure

```
.
├── data/
│   └── raw/
│       ├── churn-bigml-20.csv
│       └── churn-bigml-80.csv
├── models/
│   ├── versions/
│   ├── churn_model.joblib
│   ├── columns_order.joblib
│   ├── encoder_area.joblib
│   ├── encoder_state.joblib
│   ├── scaler.joblib
│   └── model_metadata.json
├── notebook/
│   └── Churrn.ipynb
├── elk-stack/
│   └── docker-compose.yml
├── mlartifacts/
├── mlruns/
├── app.py                    # FastAPI application
├── streamlit_app.py          # Streamlit interface
├── main.py                   # Main training script
├── evaluate_with_scores.py   # Model evaluation
├── model_pipeline.py         # ML pipeline logic
├── pipeline.py               # Data pipeline
├── version_manager.py        # Model versioning
├── elasticsearch_logger.py   # ELK logging
├── test_elasticsearch.py     # ELK testing
├── Dockerfile                # Docker configuration
├── Makefile                  # Automation scripts
├── requirements.txt          # Python dependencies
└── README.md
```

## 🛠 Installation

### Prerequisites
- Python 3.10+
- Docker and Docker Compose
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/C-Mahdi/mlops-churn-pipeline.git
   cd churn-prediction-mlops
   ```

2. **Create virtual environment**
   ```bash
   python -m venv mlops_env
   source mlops_env/bin/activate  # On Windows: mlops_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

##  Usage

### 1. Train the Model
```bash
make train
```

### 2. Evaluate the Model
```bash
make evaluate
```

### 3. Run FastAPI Server
```bash
make api
```
Or manually:
```bash
uvicorn app:app --reload
```

### 4. Run Streamlit App
```bash
make streamlit
```
Or manually:
```bash
streamlit run streamlit_app.py
```

### 5. Docker Deployment

**Build and run locally:**
```bash
make docker-build
make docker-run
```

**Push to Docker Hub:**
```bash
make docker-push
```

### 6. ELK Stack Monitoring

**Start ELK services:**
```bash
cd elk-stack
docker-compose up -d
```

**View logs in Kibana:**
- Access Kibana at http://localhost:5601
- Configure index patterns for churn-prediction logs

##  CI/CD with GitHub Actions

The project includes GitHub Actions workflows that automatically:

- Build Docker images on Dockerfile changes
- Run tests and linting
- Deploy to Docker Hub
- Trigger on push/PR to main branch

### Workflow Triggers
- Changes to `Dockerfile`
- Push to `main` branch
- Pull requests

##  Monitoring and Logging

The application integrates with the ELK stack for comprehensive monitoring:

- **Elasticsearch**: Stores logs and metrics
- **Logstash**: Processes and filters log data
- **Kibana**: Visualizes logs and creates dashboards

Logs are sent from the application using the `elasticsearch_logger.py` module.

## 🧪 Testing

Run tests:
```bash
make test
```

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


