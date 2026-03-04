from fastapi import FastAPI
from routes import router

# Initialize FastAPI application
app = FastAPI()

# Include the router from routes.py
app.include_router(router)


# Optional: Add a simple root endpoint to confirm app is running
@app.get("/app_status")
async def app_status():
    return {"status": "app is running"}
