from sqlalchemy import Column, Integer, String, DateTime, Text
from datetime import datetime
from .log_database import LogBase

class ActivityLog(LogBase):
    __tablename__ = "activity_log"

    id = Column(Integer, primary_key=True)
    agent = Column(String)
    name = Column(String)
    description = Column(Text)
    status = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
