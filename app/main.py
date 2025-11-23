from fastapi import FastAPI
from .infer import router as infer_router
from .gan_infer import router as gan_router
from .energy_infer import router as energy_router
from .diffusion_infer import router as diffusion_router

app = FastAPI(title="SPS GenAI API")


@app.get("/", tags=["health"])
def root():
    return {"message": "OK"}


app.include_router(infer_router)
app.include_router(gan_router)
app.include_router(energy_router)
app.include_router(diffusion_router)
