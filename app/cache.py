# app/cache.py

import pandas as pd
from datetime import datetime

class ASIDataCache:
    data: pd.DataFrame | None = None
    last_updated: datetime | None = None


cache = ASIDataCache()
