from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Future endpoints for sign language processing and ChatGPT interaction will go here.
