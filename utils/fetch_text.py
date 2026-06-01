import requests

from diskcache import Cache

cache = Cache("./diskcache")


@cache.memoize()
def fetch_text(url: str) -> str:
    return requests.get(url).text
