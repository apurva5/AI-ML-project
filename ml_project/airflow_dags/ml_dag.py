from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from ml_project.api.model.train import train_model
from ml_project.api.model.predict import predict_solution
from ml_project.api.model.utils import load_data

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_model_checker',
    default_args=default_args,
    description='A DAG to check for matching solutions every half hour',
    schedule_interval=timedelta(minutes=30),
)

def run_ml_model():
    model, vectorizer, packaging_encoder = train_model()
    df = load_data()

    for i, row in df.iterrows():
        problem_description = row['problem_description']
        packaging_type = row['packaging_type']
        if predict_solution(model, vectorizer, packaging_encoder, problem_description, packaging_type):
            print(f"Match found for: {problem_description}")
            print(f"GitLab Solution: {row['gitlab_solution']}")
            print(f"Technology: {row['technology']}")
            print(f"Git: {row['git']}")
            print(f"Packaging Type: {packaging_type}")
        else:
            print(f"No match found for: {problem_description}")

ml_task = PythonOperator(
    task_id='run_ml_model_task',
    python_callable=run_ml_model,
    dag=dag,
)

ml_task
