from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def hello():
    return "Hello, world!"

@app.get("/greet/{name}")
def greet(name: str):
    return f"Hello, {name}!"

@app.get("/greet_int/{num}/{st}")
def greet_int(num: int = 10, st: str = "kenneth"):
    return f"Hello, number {num}!, hoiiiii {st}"
