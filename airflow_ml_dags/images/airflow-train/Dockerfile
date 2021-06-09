FROM airflow-ml-base:latest

COPY train.py /train.py

ENTRYPOINT ["python", "train.py", '/data/processed/{{ ds }}', '/data/models/{{ ds }}']