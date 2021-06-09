import os

from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago


FEATURES_PATH = "/opt/airflow/data/raw/{{ ds }}/data.csv"
TARGET_PATH = "/opt/airflow/data/raw/{{ ds }}/target.csv"


with DAG(dag_id='train_validation_model',
         start_date=days_ago(3), schedule_interval="@weekly") as dag:
    features_sensor = FileSensor(task_id='feature sensor', poke_interval=5, retries=3, filepath=FEATURES_PATH)
    target_sensor = FileSensor(task_id='target sensor', poke_interval=5, retries=3, filepath=TARGET_PATH)

    preprocessing = DockerOperator(image='airflow-preprocess', task_id='feature preprocessing', do_xcom_push=False,
                                   command='/data/raw/{{ ds }} /data/processed/{{ ds }}',
                                   volumes=['C:/Users/anana/Desktop/MADE_2_sem/prod_ml/l3lush/airflow_ml_dags/data:/data'])
    split = DockerOperator(image='airflow-split-data', task_id='splitting date', do_xcom_push=False,
                           command='/data/processed/{{ ds }}',
                           volumes=['C:/Users/anana/Desktop/MADE_2_sem/prod_ml/l3lush/airflow_ml_dags/data:/data'])
    train = DockerOperator(image='airflow-train', task_id='train model', do_xcom_push=False,
                           command='/data/processed/{{ ds }} /data/models/{{ ds }}',
                           volumes=['C:/Users/anana/Desktop/MADE_2_sem/prod_ml/l3lush/airflow_ml_dags/data:/data'])

    validate = DockerOperator(image='airflow-validate', task_id='model validation', do_xcom_push=False,
                              command='/data/processed/{{ ds }} /data/models/{{ ds }}',
                              volumes=['C:/Users/anana/Desktop/MADE_2_sem/prod_ml/l3lush/airflow_ml_dags/data:/data'])

    log = BashOperator(task_id='report about ending', bash_command='echo "trained and validated"')

    [features_sensor, target_sensor] >> preprocessing >> split >> train >> validate >> log