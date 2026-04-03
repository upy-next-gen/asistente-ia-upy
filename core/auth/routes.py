"""HTTP routes for Microsoft Entra ID authentication."""

from typing import Any

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from core.auth.entra_id import EntraIDAuthError, EntraIDConfigError, get_entra_client
from core.utils.logger import get_logger

logger = get_logger(__name__)


async def auth_login(request: Request) -> JSONResponse:
    """Build the Entra ID authorization URL and return it as JSON.

    Args:
        request: Incoming Starlette request.

    Returns:
        A JSON response with the authentication URL, or an error payload.
    """
    del request
    try:
        client = get_entra_client()
        auth_url = client.build_auth_url()
        return JSONResponse({"auth_url": auth_url})
    except EntraIDConfigError as exc:
        logger.warning("Entra ID not configured: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=503)


async def auth_callback(request: Request) -> HTMLResponse:
    """Handle the OAuth2 callback from Microsoft Entra ID.

    Args:
        request: Incoming Starlette request containing callback query params.

    Returns:
        HTML response with the auth result.
    """
    code = request.query_params.get("code")
    error = request.query_params.get("error")

    if error:
        error_desc = request.query_params.get("error_description", error)
        logger.warning("Entra callback error: %s", error_desc)
        return HTMLResponse(
            build_auth_result_page(success=False, message=error_desc),
            status_code=400,
        )

    if not code:
        return HTMLResponse(
            build_auth_result_page(
                success=False,
                message="No se recibió código de autorización.",
            ),
            status_code=400,
        )

    try:
        client = get_entra_client()
        tokens = await client.exchange_code_for_tokens(code)
        id_token = tokens.get("id_token", "")

        if not id_token:
            return HTMLResponse(
                build_auth_result_page(
                    success=False,
                    message="No se recibió id_token.",
                ),
                status_code=400,
            )

        user = client.extract_user_from_id_token(id_token)
        logger.info("User authenticated: %s (%s)", user.name, user.email)
        return HTMLResponse(
            build_auth_result_page(
                success=True,
                message=f"¡Bienvenido, {user.name}! ({user.email})",
            )
        )
    except (EntraIDConfigError, EntraIDAuthError) as exc:
        logger.error("Auth callback failed: %s", exc)
        return HTMLResponse(
            build_auth_result_page(success=False, message=str(exc)),
            status_code=500,
        )


def build_auth_result_page(success: bool, message: str) -> str:
    """Build the HTML response shown in the auth browser tab.

    Args:
        success: Indicates whether authentication succeeded.
        message: Message shown to the user.

    Returns:
        HTML content to render in the auth callback tab.
    """
    icon = "✅" if success else "❌"
    color = "#4ade80" if success else "#f87171"
    return f"""
    <!DOCTYPE html>
    <html lang="es">
    <head><meta charset="UTF-8"><title>Autenticación</title>
    <style>
        body {{ font-family: 'Outfit', 'Segoe UI', sans-serif;
               display: flex; align-items: center; justify-content: center;
               min-height: 100vh; margin: 0;
               background: hsl(270, 47%, 7%); color: #f2f2f2; }}
        .card {{ text-align: center; padding: 2rem; }}
        .icon {{ font-size: 3rem; }}
        p {{ margin-top: 1rem; color: {color}; }}
        small {{ opacity: 0.5; }}
    </style></head>
    <body><div class="card">
        <div class="icon">{icon}</div>
        <p>{message}</p>
        <small>Esta pestaña se cerrará automáticamente en 3 segundos…</small>
    </div>
    <script>setTimeout(()=>window.close(), 3000);</script>
    </body></html>
    """


def register_auth_routes(app: Any) -> None:
    """Insert auth routes before the Chainlit catch-all route.

    Args:
        app: ASGI app exposing a mutable ``routes`` collection.
    """
    existing_paths = {getattr(route, "path", None) for route in app.routes}
    if "/auth/login" not in existing_paths:
        app.routes.insert(0, Route("/auth/login", auth_login, methods=["GET"]))
    if "/auth/callback" not in existing_paths:
        callback_index = 1 if app.routes and getattr(app.routes[0], "path", "") == "/auth/login" else 0
        app.routes.insert(
            callback_index,
            Route("/auth/callback", auth_callback, methods=["GET"]),
        )
