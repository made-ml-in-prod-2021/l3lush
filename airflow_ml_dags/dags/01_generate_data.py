from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import timedelta

default_args = {'retries': 3,
                'retry_delay': timedelta(minutes=5)}


with DAG(dag_id='generate_data', default_args=default_args, start_date=days_ago(2), schedule_interval="@daily") as dag:
    data_generate = DockerOperator(image='airflow-generate-data', command='/data/raw/{{ ds }}', network_mode="bridge",
                                   task_id='generate-data', do_xcom_push=False,
                                   volumes=['C:/Users/anana/Desktop/MADE_2_sem/prod_ml/l3lush/airflow_ml_dags/data:/data'])
