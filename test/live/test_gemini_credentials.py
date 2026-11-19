# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("google.genai")

from google.genai import Client

from ag2.live.gemini import RealTimeConfig


class TestGeminiClientAttribute:
    def test_explicit_key_reaches_client(self) -> None:
        config = RealTimeConfig("gemini-3.8-live", api_key="test-explicit-key")

        assert config.client._api_client.api_key == "test-explicit-key"

    def test_injected_client_is_reused(self) -> None:
        client = Client(api_key="test-injected-key")

        config = RealTimeConfig("gemini-3.8-live", client=client)

        assert config.client is client


@pytest.mark.parametrize("key", ["test-other-key", "test-injected-key", ""], ids=["different", "same", "empty"])
def test_client_and_key_conflict(key: str) -> None:
    client = Client(api_key="test-injected-key")

    with pytest.raises(ValueError, match="Pass either client or api_key, not both"):
        RealTimeConfig("gemini-3.8-live", client=client, api_key=key)
