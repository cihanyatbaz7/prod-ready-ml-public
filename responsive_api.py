from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from typing import List

app = FastAPI()

class User(BaseModel):
    id: int
    name: str = "John Doe"
    signup_ts: datetime | None = None
    friend_ids: List[int] | None = None
    password: str

@app.post("/user")
async def read_user(user: User) -> User:
    return user

@app.post("/failing_user")
async def read_user(user: User) -> User:
    # the program generates invalid data, i.e. missing a password
    return {"id":user.id, "name":"John Doe"}

class SafeUser(BaseModel):
    id: int
    name: str = "John Doe"
    friend_ids: List[int] | None = None


@app.post("/safe_user")
async def read_user(user: User) -> SafeUser:
    return user