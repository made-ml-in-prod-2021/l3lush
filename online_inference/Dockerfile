FROM python:3.9
COPY requirements.txt ./requirements.txt
COPY setup.py ./setup.py
RUN python setup.py install
RUN pip install -r requirements.txt

COPY models /models
COPY src /src
COPY configs /configs
COPY data /data

WORKDIR .

ENV PATH_TO_MODEL="/model_forest/pkl"

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]