from typing import List
from pydantic import BaseModel, Field

SEQ_LEN = 72

class PM25Request(BaseModel):
    nitrogen_dioxide: List[float] = Field(..., min_items=SEQ_LEN, max_items=SEQ_LEN)
    temperature_2m: List[float] = Field(..., min_items=SEQ_LEN, max_items=SEQ_LEN)
    relative_humidity_2m: List[float] = Field(..., min_items=SEQ_LEN, max_items=SEQ_LEN)
    wind_speed_10m: List[float] = Field(..., min_items=SEQ_LEN, max_items=SEQ_LEN)
    wind_direction_10m: List[float] = Field(..., min_items=SEQ_LEN, max_items=SEQ_LEN)
    precipitation: List[float] = Field(..., min_items=SEQ_LEN, max_items=SEQ_LEN)
    surface_pressure: List[float] = Field(..., min_items=SEQ_LEN, max_items=SEQ_LEN)
    hour: List[int]   = Field(..., min_items=SEQ_LEN, max_items=SEQ_LEN)
    day_of_week: List[int]   = Field(..., min_items=SEQ_LEN, max_items=SEQ_LEN)
    month: List[int]   = Field(..., min_items=SEQ_LEN, max_items=SEQ_LEN)
    is_weekend: List[int]   = Field(..., min_items=SEQ_LEN, max_items=SEQ_LEN)
    wind_u: List[float] = Field(..., min_items=SEQ_LEN, max_items=SEQ_LEN)
    wind_v: List[float] = Field(..., min_items=SEQ_LEN, max_items=SEQ_LEN)
    is_rainy: List[int]   = Field(..., min_items=SEQ_LEN, max_items=SEQ_LEN)
