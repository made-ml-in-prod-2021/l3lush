FROM airflow-ml-base:latest

COPY preprocess.py /preprocess.py

ENTRYPOINT ["python", "preprocess.py", '/data/raw/{{ ds }}', '/data/processed/{{ ds }}']