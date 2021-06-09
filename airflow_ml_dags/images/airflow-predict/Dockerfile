FROM airflow-ml-base:latest

COPY predict.py /predict.py

ENTRYPOINT ["python", "predict.py",'/data/processed/{{ ds }}' ,'/data/models/{{ ds }}']