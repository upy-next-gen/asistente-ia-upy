from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from core.auth.routes import auth_callback, auth_login, build_auth_result_page, register_auth_routes


@pytest.mark.asyncio
@patch("core.auth.routes.get_entra_client")
async def test_auth_login_returns_auth_url(mock_get_entra_client):
    client = MagicMock()
    client.build_auth_url.return_value = "https://login.microsoftonline.com/auth"
    mock_get_entra_client.return_value = client

    response = await auth_login(MagicMock())
    assert response.status_code == 200
    assert b'"auth_url":"https://login.microsoftonline.com/auth"' in response.body


@pytest.mark.asyncio
@patch("core.auth.routes.get_entra_client")
async def test_auth_callback_success(mock_get_entra_client):
    client = MagicMock()
    client.exchange_code_for_tokens = AsyncMock(return_value={"id_token": "header.payload.sig"})
    client.extract_user_from_id_token.return_value = MagicMock(name="Nombre", email="mail@test.com")
    client.extract_user_from_id_token.return_value.name = "Nombre"
    client.extract_user_from_id_token.return_value.email = "mail@test.com"
    mock_get_entra_client.return_value = client

    request = MagicMock()
    request.query_params = {"code": "abc123"}

    response = await auth_callback(request)
    assert response.status_code == 200
    assert "¡Bienvenido, Nombre! (mail@test.com)" in response.body.decode("utf-8")


@pytest.mark.asyncio
async def test_auth_callback_missing_code():
    request = MagicMock()
    request.query_params = {}

    response = await auth_callback(request)
    assert response.status_code == 400
    assert "No se recibió código de autorización." in response.body.decode("utf-8")


def test_build_auth_result_page_mentions_tab():
    html = build_auth_result_page(success=True, message="ok")
    assert "Esta pestaña se cerrará automáticamente en 3 segundos" in html


def test_register_auth_routes_inserts_before_existing_routes():
    app = Starlette(routes=[Route("/", endpoint=lambda request: None)])
    register_auth_routes(app)
    assert app.routes[0].path == "/auth/login"
    assert app.routes[1].path == "/auth/callback"
