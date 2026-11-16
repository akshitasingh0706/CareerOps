import json
from pathlib import Path

import httpx
import pytest

FIX = Path(__file__).parent / "fixtures"


def load(name: str):
    return json.loads((FIX / name).read_text())


@pytest.fixture
def mock_client():
    """httpx client that serves fixtures by host and never touches the network."""
    routes = {
        "boards-api.greenhouse.io": "greenhouse_acme.json",
        "api.lever.co": "lever_beta.json",
        "api.ashbyhq.com": "ashby_gamma.json",
    }
    state = {"requests": [], "override": {}}

    def handler(request: httpx.Request) -> httpx.Response:
        state["requests"].append(str(request.url))
        host = request.url.host
        if host in state["override"]:
            return state["override"][host]
        return httpx.Response(200, json=load(routes[host]))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    client.state = state
    return client
