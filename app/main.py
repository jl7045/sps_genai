from fastapi import FastAPI
from .infer import router as infer_router
from .gan_infer import router as gan_router

app = FastAPI(title="SPS GenAI API")

@app.get("/", tags=["health"])
def root():
    return {"message": "OK"}

app.include_router(infer_router)
app.include_router(gan_router)
<<<<<<< HEAD
=======

>>>>>>> 5eba60f6627c0b39a9a28073b2ea2c69feffe620
