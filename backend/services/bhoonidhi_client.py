from __future__ import annotations

import requests


class BhoonidhiClient:
    """
    Client for communicating with the Bhoonidhi API.
    """

    BASE_URL = "https://bhoonidhi-api.nrsc.gov.in"

    def __init__(self):

        self.session = requests.Session()

        self.access_token = None

        self.refresh_token = None

    #####################################################################
    # AUTHENTICATION
    #####################################################################

    def login(self, user_id: str, password: str):

        url = f"{self.BASE_URL}/auth/token"

        payload = {
            "userId": user_id,
            "password": password,
            "grant_type": "password",
        }

        response = self.session.post(
            url,
            json=payload,
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        self.access_token = data["access_token"]

        self.refresh_token = data["refresh_token"]

        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.access_token}"
            }
        )

        return data

    #####################################################################
    # REFRESH TOKEN
    #####################################################################

    def refresh(self):

        url = f"{self.BASE_URL}/auth/refresh-token"

        payload = {
            "refresh_token": self.refresh_token,
            "grant_type": "refresh_token",
        }

        response = self.session.post(
            url,
            json=payload,
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        self.access_token = data["access_token"]

        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.access_token}"
            }
        )

        return data

    #####################################################################
    # COLLECTIONS
    #####################################################################

    def collections(self):

        url = f"{self.BASE_URL}/data/collections"

        response = self.session.get(url)

        response.raise_for_status()

        return response.json()

    #####################################################################
    # SEARCH
    #####################################################################

    def search(self, payload: dict):

        url = f"{self.BASE_URL}/data/search"

        response = self.session.post(
            url,
            json=payload,
            timeout=120,
        )

        response.raise_for_status()

        return response.json()

    #####################################################################
    # DOWNLOAD
    #####################################################################

    def download(
        self,
        item_id: str,
        collection: str,
        save_path: str,
    ):

        url = (
            f"{self.BASE_URL}/download"
            f"?id={item_id}"
            f"&collection={collection}"
        )

        response = self.session.get(
            url,
            stream=True,
        )

        response.raise_for_status()

        with open(save_path, "wb") as f:

            for chunk in response.iter_content(1024 * 1024):

                if chunk:

                    f.write(chunk)

        return save_path