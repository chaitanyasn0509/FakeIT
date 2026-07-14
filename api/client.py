from dataclasses import dataclass
import requests


@dataclass
class BhoonidhiClient:
    base_url: str
    api_key: str

    def __post_init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}"
        })

    def search(self, **kwargs):
        raise NotImplementedError

    def download(self, scene_id):
        raise NotImplementedError

    def metadata(self, scene_id):
        raise NotImplementedError