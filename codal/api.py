from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# TODO: Make it a public func
from .cli import _ask
from .database import SessionLocal

app = FastAPI()

origins = [
    "http://localhost:3000",  # React app
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/ask/{question}")
# TODO: Make repo a param
async def ask(question: str, repo: str = "seem/codal", db: Session = Depends(get_db)):
    answer = _ask(repo, question, db)
    return answer
