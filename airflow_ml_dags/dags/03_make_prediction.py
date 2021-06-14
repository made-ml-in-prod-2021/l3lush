from airflow.utils.dates import days_ago
from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator

from datetime import timedelta


DEFAULT_MODEL_PATH = "/opt/airflow/data/models/{{ ds }}/model.pkl"
DEFAULT_DATA_PATH = "/opt/airflow/data/processed/{{ ds }}/test.csv"
DEFAULT_SCALER_PATH = "/opt/airflow/data/processed/{{ ds }}/scaler.csv"

default_args = {'retries': 3,
                'retry_delay': timedelta(minutes=5)}

with DAG(dag_id='prediction', default_args=default_args, start_date=days_ago(3), schedule_interval="@daily") as dag:
    log = BashOperator(task_id='report-ls', bash_command='echo $(ls)')
    log_2 = BashOperator(task_id='report-folder', bash_command='echo $(ls /opt/airflow/data/models/)')
    data_sensor = FileSensor(task_id='data-sensor', poke_interval=5, retries=3, filepath=DEFAULT_DATA_PATH)
    model_sensor = FileSensor(task_id='model-sensor', poke_interval=5, retries=3, filepath=DEFAULT_MODEL_PATH)
    scaler_sensor = FileSensor(task_id='scaler-sensor', poke_interval=5, retries=3, filepath=DEFAULT_SCALER_PATH)

    prediction = DockerOperator(image='airflow-predict',
                                command="/data/processed/{{ ds } /data/models/{{ ds }}",
                                network_mode="bridge",
                                task_id='prediction', do_xcom_push=False,
                                volumes=['C:/Users/anana/Desktop/MADE_2_sem/prod_ml/l3lush/airflow_ml_dags/data:/data'])

    [data_sensor, model_sensor, scaler_sensor] >> prediction
