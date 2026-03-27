"""Microsoft Entra ID OAuth2 client.

Handles the OAuth2 authorization code flow for Microsoft Entra ID:
- Building the authorization URL
- Exchanging the authorization code for tokens
- Validating the ID token (basic JWT validation)

This module does NOT implement PKCE or nonce validation — those are
planned for a future iteration.
"""

import urllib.parse
from dataclasses import dataclass

import httpx

from core.config import settings
from core.utils.logger import get_logger

logger = get_logger(__name__)

ENTRA_AUTHORITY = "https://login.microsoftonline.com"


@dataclass
class EntraIDUser:
    """Represents an authenticated Entra ID user."""

    sub: str
    email: str
    name: str


class EntraIDConfigError(Exception):
    """Raised when Entra ID environment variables are not configured."""


class EntraIDAuthError(Exception):
    """Raised when an authentication step fails."""


class EntraIDClient:
    """OAuth2 client for Microsoft Entra ID authorization code flow.

    Args:
        tenant_id: Azure AD tenant ID.
        client_id: Application (client) ID registered in Entra.
        client_secret: Client secret for the application.
        redirect_uri: Registered redirect URI for the OAuth callback.
    """

    SCOPES = "openid email profile"

    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
    ) -> None:
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self._authorize_url = (
            f"{ENTRA_AUTHORITY}/{tenant_id}/oauth2/v2.0/authorize"
        )
        self._token_url = (
            f"{ENTRA_AUTHORITY}/{tenant_id}/oauth2/v2.0/token"
        )

    def build_auth_url(self, state: str = "") -> str:
        """Build the Microsoft authorization URL.

        Args:
            state: Optional opaque value to maintain state between
                   the request and callback.

        Returns:
            The full authorization URL to redirect the user to.
        """
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": self.SCOPES,
            "response_mode": "query",
        }
        if state:
            params["state"] = state

        return f"{self._authorize_url}?{urllib.parse.urlencode(params)}"

    async def exchange_code_for_tokens(self, code: str) -> dict:
        """Exchange the authorization code for access and ID tokens.

        Args:
            code: The authorization code received from Entra callback.

        Returns:
            Dict with ``access_token``, ``id_token``, and ``token_type``.

        Raises:
            EntraIDAuthError: If the token exchange fails.
        """
        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code",
            "scope": self.SCOPES,
        }

        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(self._token_url, data=payload)

        if response.status_code != 200:
            logger.error(
                "Token exchange failed: %s %s",
                response.status_code,
                response.text,
            )
            raise EntraIDAuthError(
                f"Token exchange failed with status {response.status_code}"
            )

        return response.json()

    @staticmethod
    def extract_user_from_id_token(id_token: str) -> EntraIDUser:
        """Extract user information from the ID token (basic decode).

        Performs a base64 decode of the JWT payload **without**
        cryptographic signature verification.  Full RS256 validation
        is planned for a future iteration.

        Args:
            id_token: The JWT ID token string.

        Returns:
            An ``EntraIDUser`` with sub, email, and name.

        Raises:
            EntraIDAuthError: If the token cannot be decoded.
        """
        import base64
        import json

        try:
            # JWT has 3 parts: header.payload.signature
            payload_b64 = id_token.split(".")[1]
            # Add padding
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += "=" * padding

            payload_bytes = base64.urlsafe_b64decode(payload_b64)
            claims = json.loads(payload_bytes)

            return EntraIDUser(
                sub=claims.get("sub", ""),
                email=claims.get("preferred_username", claims.get("email", "")),
                name=claims.get("name", ""),
            )
        except (IndexError, ValueError, json.JSONDecodeError) as exc:
            logger.error("Failed to decode id_token: %s", exc)
            raise EntraIDAuthError("Invalid id_token format") from exc


def get_entra_client() -> EntraIDClient:
    """Factory function that creates an EntraIDClient from settings.

    Raises:
        EntraIDConfigError: If any required Entra env var is missing.
    """
    missing = [
        var
        for var in (
            "ENTRA_TENANT_ID",
            "ENTRA_CLIENT_ID",
            "ENTRA_CLIENT_SECRET",
            "ENTRA_REDIRECT_URI",
        )
        if not getattr(settings, var, "")
    ]
    if missing:
        raise EntraIDConfigError(
            f"Entra ID no está configurado. Variables faltantes: {missing}"
        )

    return EntraIDClient(
        tenant_id=settings.ENTRA_TENANT_ID,
        client_id=settings.ENTRA_CLIENT_ID,
        client_secret=settings.ENTRA_CLIENT_SECRET,
        redirect_uri=settings.ENTRA_REDIRECT_URI,
    )
