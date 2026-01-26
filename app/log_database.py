from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

LOG_DATABASE_URL = "postgresql://neondb_owner:npg_8ksy5LGPnjJD@ep-quiet-glade-a18rsovg-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

log_engine = create_engine(LOG_DATABASE_URL, pool_pre_ping=True)
LogSessionLocal = sessionmaker(bind=log_engine)
LogBase = declarative_base()
