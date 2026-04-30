from fastapi import FastAPI
import time
import asyncio

app = FastAPI()
 
@app.get("/sleep/{seconds}")
def sleep_for(seconds: int):
    time.sleep(seconds)
    return "Awake!"


@app.get("/sleepio/{seconds}")
async def sleep_for(seconds: int):
    print("Going to bed")
    for s in range(seconds+1):
      await asyncio.sleep(1)
      print(f"zz {s}")
    return "Awake!"