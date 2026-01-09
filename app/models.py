# app/models.py

from sqlalchemy import Column, Integer, Float, String, Boolean, Date, DateTime
from datetime import datetime
from .database import Base

class ASIAlert(Base):
    __tablename__ = "asi_alerts"

    id = Column(Integer, primary_key=True, index=True)
    run_date = Column(Date, index=True)
    vol_regime = Column(Boolean)

    prob_minus_5 = Column(Float)
    prob_minus_10 = Column(Float)
    prob_minus_20 = Column(Float)
    prob_plus_5 = Column(Float)
    prob_plus_10 = Column(Float)
    prob_plus_20 = Column(Float)

    alert_minus_5 = Column(String)
    alert_minus_10 = Column(String)
    alert_minus_20 = Column(String)

    created_at = Column(DateTime, default=datetime.utcnow)
