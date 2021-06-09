from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.docker.operators.docker import DockerOperator


with DAG(dag_id='generate_data', start_date=days_ago(3), schedule_interval="@daily") as dag:
    data_generate = DockerOperator(image='airflow-generate-data',
                                   task_id='generate_data', do_xcom_push=False,
                                   volumes=['C:/Users/anana/Desktop/MADE_2_sem/prod_ml/l3lush/airflow_ml_dags/data:/data'])
