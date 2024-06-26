# server.py
import os
import sys
import traceback

from typing import Tuple, List
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.logger import logger
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseSettings, BaseModel

from model import Model, get_model

class Settings(BaseSettings):
    BASE_URL = "http://localhost:8000"
    USE_NGROK = os.environ.get("USE_NGROK", "False") == "True"


settings = Settings()


# Initialize the FastAPI app for a simple web server
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="")

if settings.USE_NGROK:
  
  # pyngrok should only ever be installed or initialized in a dev environment when this flag is set
  from pyngrok import ngrok

  # Get the dev server port (defaults to 8000 for Uvicorn, can be overridden with `--port`
  # when starting the server
  port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else 8000

  # Open a ngrok tunnel to the dev server
  public_url = ngrok.connect(port)

  logger.warn("ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}/\"".format(public_url, port))

  # Update any base URLs or webhooks to use the public ngrok URL
  settings.BASE_URL = public_url
  
class DistanceRequest(BaseModel):
  sents: Tuple[str, str]

class DistanceResponse(BaseModel):
  cosine: float
  manhattan: float
  euclidean: float

class KmeansRequest(BaseModel):
  corpus: List[str]
  n_clusters: int

class KmeansResponse(BaseModel):
  plot_html: str

@app.get("/")
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "haha": "hhi", "serveradd": request.url_for('root')})

@app.post("/measure_dis", response_model=DistanceResponse)
def measure_sim(request: DistanceRequest, model:Model = Depends(get_model)):
  try:
    cosine, manhattan, euclidean = model.measure_distance(request.sents)
  except Exception as e:
    raise HTTPException(
          status_code=422,
          detail=traceback.format_exc(),
      )

  return DistanceResponse(
      cosine = cosine,
      manhattan = manhattan,
      euclidean = euclidean
  )

@app.post("/cluster", response_model=KmeansResponse)
def cluster(request: KmeansRequest, model:Model = Depends(get_model)):
  print(request.n_clusters, len(request.corpus))
  if (request.n_clusters < 1):
    raise HTTPException(status_code=400,
                  detail="Number of clusters must be greater than 0.")
  elif (len(request.corpus) < 2):
    raise HTTPException(status_code=400,
                  detail="Corpus must have at least 2 sentences.")
  elif (len(request.corpus) < request.n_clusters):
    raise HTTPException(status_code=400,
                  detail="Number of sentences must be greater than number of clusters.")
  try:
    result = KmeansResponse(
      plot_html = model.fit_kmeans(request.corpus, request.n_clusters)
    )
  except Exception as e:
    raise HTTPException(
            status_code=422,
            detail=traceback.format_exc(),
          )
  return result