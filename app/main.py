# app/main.py

from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from .database import SessionLocal, engine
from .models import Base, ASIAlert
from .scheduler import start_scheduler

Base.metadata.create_all(bind=engine)

app = FastAPI(title="ASI Risk Alert Service")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
def startup():
    start_scheduler()

@app.get("/alerts/latest")
def latest_alert(db: Session = Depends(get_db)):
    return db.query(ASIAlert).order_by(ASIAlert.run_date.desc()).first()

@app.get("/alerts/history")
def alert_history(limit: int = 30, db: Session = Depends(get_db)):
    return (
        db.query(ASIAlert)
        .order_by(ASIAlert.run_date.desc())
        .limit(limit)
        .all()
    )

@app.get("/health")
def health():
    return {"status": "ok"}
