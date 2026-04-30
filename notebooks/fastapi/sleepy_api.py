from fastapi import FastAPI
import time

app = FastAPI()
 
@app.get("/sleep/{seconds}")
def sleep_for(seconds: int):
    time.sleep(seconds)
    return "Awake!"