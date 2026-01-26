# app/logger.py

from app.log_database import LogSessionLocal
from app.log_models import ActivityLog
from datetime import datetime
from sqlalchemy import text


def write_log(name, description, status):
    db = LogSessionLocal()
    query = text("""
        INSERT INTO activity_log
        (
            label,
            name,
            description,
            timestamp,
            agent
        )
        VALUES
        (
            :label,
            :name,
            :description,
            :timestamp,
            :agent
        )
    """)
    
    db.execute(query, {
        "label": name,
        "name": status,  
        "description": description,
        "timestamp": datetime.now(),
        "agent": "ASI_RISK_ENGINE"
    })
    
    db.commit()
    

