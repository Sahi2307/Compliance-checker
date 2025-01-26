from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.responses import HTMLResponse
from app.routes.upload import router as upload_router
from app.routes.analyze import router as analyze_router

# Create FastAPI app instance
app = FastAPI()

# Setup Jinja2 templates for HTML rendering
templates = Jinja2Templates(directory="app/templates")

# Serve static files (CSS, JS, Images)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include routes from upload.py and analyze.py
app.include_router(upload_router)
app.include_router(analyze_router)

# Home route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

