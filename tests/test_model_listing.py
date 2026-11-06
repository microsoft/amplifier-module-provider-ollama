"""Real Ollama SDK transport verifies exact installed IDs and truthful failures."""

import asyncio
import json

import httpx
import pytest
from ollama import AsyncClient
from amplifier_core.llm_errors import AuthenticationError, LLMError
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_ollama import OllamaProvider
from amplifier_module_provider_ollama._model_listing import ModelFilters

IDS = ["Team/Custom:7b-q4", "Team/Custom:70b-q8", "production", "future-2099-01-01"]


def make(handler, **config):
    p = OllamaProvider(
        host="https://ollama.invalid", config={"max_retries": 0, **config}
    )
    p._client = AsyncClient(
        host="https://ollama.invalid",
        headers={"X-Tenant": "a"},
        transport=httpx.MockTransport(handler),
    )
    return p


@pytest.mark.asyncio
async def test_all_tags_filters_refresh_and_exact_inference():
    calls = []
    cfg = {"include": ["Team/*"]}

    def handler(request):
        calls.append(request)
        assert request.headers["x-tenant"] == "a"
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"model": i} for i in IDS]})
        if request.url.path == "/api/show":
            return httpx.Response(
                200, json={"model_info": {"llama.context_length": 8192}}
            )
        body = json.loads(request.content)
        assert body["model"] == "hidden:Q8"
        assert body["options"]["top_p"] == 0.4
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "message": {"role": "assistant", "content": "ok"},
                "done": True,
                "prompt_eval_count": 1,
                "eval_count": 1,
            },
        )

    p = make(handler, model_filters=cfg, use_streaming=False, top_p=0.4)
    cfg["include"].clear()
    try:
        assert [m.id for m in await p.list_models()] == sorted(IDS[:2])
        assert [m.id for m in await p.list_models(filters=None)] == sorted(IDS[:2])
        assert [m.id for m in await p.list_models(filters={})] == sorted(IDS)
        assert [
            m.id
            for m in await p.list_models(
                filters={"include": ["TEAM/*"], "exclude": ["*:70B-*"]}
            )
        ] == [IDS[0]]
        assert await p.list_models(filters={"include": []}) == []
        assert all(
            m.context_window == 0 and m.max_output_tokens == 0 and not m.capabilities
            for m in await p.list_models()
        )
        info = p.get_model_list_info()
        assert info["source"] == "provider-api"
        info["default_filters"]["include"].clear()
        assert p.get_model_list_info()["default_filters"]["include"] == ["Team/*"]
        await p.complete(
            ChatRequest(
                model="hidden:Q8", messages=[Message(role="user", content="test")]
            ),
            model="hidden:Q8",
        )
        assert any(r.url.path == "/api/chat" for r in calls)
    finally:
        await p._client._client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [{}, {"models": None}, {"models": [{"model": ""}]}, {"models": [{"model": 5}]}],
)
async def test_malformed_not_empty(payload):
    p = make(lambda r: httpx.Response(200, json=payload))
    try:
        with pytest.raises(LLMError):
            await p.list_models()
    finally:
        await p._client._client.aclose()


@pytest.mark.asyncio
async def test_success_empty_error_and_cancel_distinct():
    responses = [
        httpx.Response(200, json={"models": []}),
        httpx.Response(401, json={"error": "SECRET"}),
    ]
    p = make(lambda r: responses.pop(0))
    try:
        assert await p.list_models() == []
        with pytest.raises(AuthenticationError) as caught:
            await p.list_models()
        assert "SECRET" not in str(caught.value)
    finally:
        await p._client._client.aclose()

    async def cancel(r):
        raise asyncio.CancelledError()

    p = make(cancel)
    try:
        with pytest.raises(asyncio.CancelledError):
            await p.list_models()
    finally:
        await p._client._client.aclose()


@pytest.mark.parametrize(
    "value",
    [None, [], {"include": "*"}, {"exclude": [1]}, {"hide_dated": 1}, {"unknown": []}],
)
def test_invalid_filters(value):
    with pytest.raises((TypeError, ValueError)):
        OllamaProvider(config={"model_filters": value})


def test_common_vectors():
    fixture = ["Custom:Q4", "Custom:Q8", "future", "future-2099-01-01"]
    assert ModelFilters.parse({}).select(fixture) == sorted(fixture)
    assert ModelFilters.parse({"include": ["CUSTOM:*"], "exclude": ["*:q8"]}).select(
        fixture
    ) == ["Custom:Q4"]
    assert ModelFilters.parse({"hide_dated": True}).select(fixture) == [
        "Custom:Q4",
        "Custom:Q8",
        "future",
    ]
    assert ModelFilters.parse({"include": []}).select(fixture) == []
