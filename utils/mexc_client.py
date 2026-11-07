"""
MEXC Public API Client
Only uses real-time ticker for current price.
Klines are used ONLY for bootstrapping historical data at first run.
After that, we rely on stored ticker snapshots.
"""

import requests
import time
from typing import Optional, List, Dict
import config

class MexcClient:
    def __init__(self):
        self.base_url = config.MEXC_BASE_URL
        self.symbol = config.SYMBOL
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "BTC-Hybrid-Predictor/1.0"
        })

    def get_ticker_price(self) -> Optional[float]:
        """
        Get REAL-TIME last price (ticker) from MEXC Spot.
        Endpoint: GET /api/v3/ticker/price?symbol=BTCUSDT
        This is the live ticker price, NOT a closed candle.
        """
        url = f"{self.base_url}/api/v3/ticker/price"
        params = {"symbol": self.symbol}
        try:
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            price = float(data["price"])
            return price
        except Exception as e:
            print(f"[MEXC] Error fetching ticker: {e}")
            return None

    def get_ticker_24hr(self) -> Optional[Dict]:
        """More detailed 24hr ticker (still live)."""
        url = f"{self.base_url}/api/v3/ticker/24hr"
        params = {"symbol": self.symbol}
        try:
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[MEXC] Error fetching 24hr ticker: {e}")
            return None

    def get_klines(self, interval: str = "1m", limit: int = 500) -> List[List]:
        """
        Used ONLY for initial bootstrap of historical data.
        After the system has enough live ticker history, we stop relying on this.
        """
        url = f"{self.base_url}/api/v3/klines"
        params = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": limit
        }
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[MEXC] Error fetching klines: {e}")
            return []

    def get_server_time(self) -> Optional[int]:
        url = f"{self.base_url}/api/v3/time"
        try:
            resp = self.session.get(url, timeout=5)
            resp.raise_for_status()
            return resp.json().get("serverTime")
        except Exception:
            return None
