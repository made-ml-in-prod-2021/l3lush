FROM airflow-ml-base:latest

COPY make_dataset.py /make_dataset.py

ENTRYPOINT ["python", "make_dataset.py", '/data/raw/{{ ds }}']