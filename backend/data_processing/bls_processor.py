import aiohttp
import asyncio
from typing import Dict, List, Any
import logging
from datetime import datetime

class BLSProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.bls.gov/publicAPI/v2"
        self.headers = {
            "BLS-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        
    async def get_occupation_data(self, occupation_code: str) -> Dict[str, Any]:
        """Fetch occupation data from BLS API"""
        async with aiohttp.ClientSession(headers=self.headers) as session:
            url = f"{self.base_url}/timeseries/data/"
            payload = {
                "seriesid": [occupation_code],
                "startyear": str(datetime.now().year - 5),
                "endyear": str(datetime.now().year)
            }
            
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logging.error(f"Error fetching BLS data: {response.status}")
                    return None
    
    async def get_occupation_outlook(self, occupation_code: str) -> Dict[str, Any]:
        """Fetch occupation outlook data"""
        async with aiohttp.ClientSession(headers=self.headers) as session:
            url = f"{self.base_url}/occupation/outlook/{occupation_code}"
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logging.error(f"Error fetching outlook data: {response.status}")
                    return None
    
    async def get_industry_data(self, industry_code: str) -> Dict[str, Any]:
        """Fetch industry data"""
        async with aiohttp.ClientSession(headers=self.headers) as session:
            url = f"{self.base_url}/industry/{industry_code}"
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logging.error(f"Error fetching industry data: {response.status}")
                    return None
    
    async def get_salary_data(self, occupation_code: str) -> Dict[str, Any]:
        """Fetch salary data for an occupation"""
        async with aiohttp.ClientSession(headers=self.headers) as session:
            url = f"{self.base_url}/wages/occupation/{occupation_code}"
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logging.error(f"Error fetching salary data: {response.status}")
                    return None 