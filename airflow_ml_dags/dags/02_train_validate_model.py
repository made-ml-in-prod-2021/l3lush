import os

from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from datetime import timedelta


FEATURES_PATH = "/opt/airflow/data/raw/{{ ds }}/data.csv"
TARGET_PATH = "/opt/airflow/data/raw/{{ ds }}/target.csv"

default_args = {'retries': 3,
                'retry_delay': timedelta(minutes=5)}


with DAG(dag_id='train_validation_model',
            default_args=default_args,
         start_date=days_ago(3), schedule_interval="@weekly") as dag:
    print(os.listdir())
    features_sensor = FileSensor(task_id='feature_sensor', poke_interval=5, retries=3, filepath=FEATURES_PATH)
    target_sensor = FileSensor(task_id='target_sensor', poke_interval=5, retries=3, filepath=TARGET_PATH)

    preprocessing = DockerOperator(image='airflow-preprocess',
                                   command="/data/raw/{{ ds }} /data/processed/{{ ds }}",
                                   network_mode="bridge",
                                   task_id='feature_preprocessing', do_xcom_push=False,
                                   volumes=['C:/Users/anana/Desktop/MADE_2_sem/prod_ml/l3lush/airflow_ml_dags/data:/data'])
    split = DockerOperator(image='airflow-split-data', command="/data/processed/{{ ds }}",
                           network_mode="bridge",
                           task_id='splitting_date', do_xcom_push=False,
                           volumes=['C:/Users/anana/Desktop/MADE_2_sem/prod_ml/l3lush/airflow_ml_dags/data:/data'])
    train = DockerOperator(image='airflow-train', task_id='train_model', do_xcom_push=False,
                           command="/data/processed/{{ ds }} /data/models/{{ ds }}",
                           network_mode="bridge",
                           volumes=['C:/Users/anana/Desktop/MADE_2_sem/prod_ml/l3lush/airflow_ml_dags/data:/data'])

    validate = DockerOperator(image='airflow-validate', task_id='model_validation', do_xcom_push=False,
                              command="/data/processed/{{ ds }} /data/models/{{ ds }}",
                              network_mode="bridge",
                              volumes=['C:/Users/anana/Desktop/MADE_2_sem/prod_ml/l3lush/airflow_ml_dags/data:/data'])

    log = BashOperator(task_id='report_about_ending', bash_command='echo "trained and validated"')
    [features_sensor, target_sensor] >> preprocessing >> split >> train >> validate >> log
