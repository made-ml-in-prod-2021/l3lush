FROM airflow-ml-base:latest

COPY validate.py /validate.py

ENTRYPOINT ["python", "validate.py", '/data/processed/{{ ds }}', '/data/models/{{ ds }}']