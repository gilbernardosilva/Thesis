import requests

from app.core.config import settings
from app.schemas.location import Location, LocationSTG


def fetch_road_info(
    locations: list[Location],
):
    endpoint = "/v1/roads/info"
    url = f"{settings.api_prod_url}{endpoint}?access_token={settings.api_prod_access_token}"
    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "language": "en",
        "locations": [loc.model_dump() for loc in locations],
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        raise


def fetch_off_route_info(
    locations: list[LocationSTG],
):
    endpoint = "/v1/roads/off_route_info"
    url = (
        f"{settings.api_stg_url}{endpoint}?access_token={settings.api_stg_access_token}"
    )
    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "language": "en",
        "locations": [loc.model_dump() for loc in locations],
        "safety_cameras": False,
        "speed_limit": True,
        "timeout": 0.5,
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        raise
