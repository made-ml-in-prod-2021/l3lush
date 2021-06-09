from airflow.utils.dates import days_ago
from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.providers.docker.operators.docker import DockerOperator


DEFAULT_MODEL_PATH = '/opt/airflow/data/models/{{ ds }}/model.pkl'
DEFAULT_DATA_PATH = '/opt/airflow/data/processed/{{ ds }}/test.csv'
DEFAULT_SCALER_PATH = '/opt/airflow/data/processed/{{ ds }}/scaler.csv'


with DAG(dag_id='prediction', start_date=days_ago(3), schedule_interval="@daily") as dag:
    data_sensor = FileSensor(task_id='datasensor', poke_interval=3, retries=3, filepath=DEFAULT_DATA_PATH)
    model_sensor = FileSensor(task_id='modelsensor', poke_interval=3, retries=3, filepath=DEFAULT_MODEL_PATH)
    scaler_sensor = FileSensor(task_id='scalersensor', poke_interval=3, retries=3, filepath=DEFAULT_SCALER_PATH)

    prediction = DockerOperator(image='airflow-predict', task_id='prediction', do_xcom_push=False,
                                volumes=['C:/Users/anana/Desktop/MADE_2_sem/prod_ml/l3lush/airflow_ml_dags/data:/data'])

    [data_sensor, model_sensor, scaler_sensor] >> prediction
