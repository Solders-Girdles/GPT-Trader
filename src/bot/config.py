from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class AlpacaSettings(BaseModel):
    api_key_id: str | None = os.getenv("ALPACA_API_KEY_ID")
    api_secret_key: str | None = os.getenv("ALPACA_API_SECRET_KEY")
    paper_base_url: str = os.getenv("ALPACA_PAPER_BASE_URL", "https://paper-api.alpaca.markets")


class Settings(BaseModel):
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    alpaca: AlpacaSettings = AlpacaSettings()


settings = Settings()
