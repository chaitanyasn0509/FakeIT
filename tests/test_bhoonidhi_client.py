"""Tests for the Bhoonidhi API client."""

from __future__ import annotations

import time

from datasets.bhoonidhi_client import BhoonidhiClient, BhoonidhiSettings


def test_token_response_updates_state() -> None:
    """Token payloads are parsed into reusable token state."""
    client = BhoonidhiClient(
        BhoonidhiSettings(
            base_url="https://bhoonidhi-api.nrsc.gov.in",
            username="user",
            password="pass",
        )
    )
    try:
        client._update_token(
            {
                "access_token": "access",
                "refresh_token": "refresh",
                "token_type": "Bearer",
                "expires_in": 1200,
            }
        )
        assert client.token.access_token == "access"
        assert client.token.refresh_token == "refresh"
        assert client.token.expires_at > time.time()
    finally:
        client.close()


def test_online_filter_matches_bhoonidhi_cql2() -> None:
    """The online-only filter matches the official CQL2 shape."""
    client = BhoonidhiClient(
        BhoonidhiSettings(
            base_url="https://bhoonidhi-api.nrsc.gov.in",
            username="user",
            password="pass",
        )
    )
    try:
        assert client.online_filter() == {"args": [{"property": "Online"}, "Y"], "op": "eq"}
    finally:
        client.close()
