import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def main():
    return "it is entry point of our predictor"


if __name__ == "__main__":
    uvicorn.run("app:app", host='0.0.0.0', port=8888)
