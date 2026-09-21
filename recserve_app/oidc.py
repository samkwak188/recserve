"""Google authorization code + PKCE. All identity validation uses Authlib."""
import time
import httpx
from authlib.integrations.httpx_client import AsyncOAuth2Client
from authlib.jose import JsonWebToken
from authlib.oidc.core import CodeIDToken

AUTHORIZE = 'https://accounts.google.com/o/oauth2/v2/auth'
TOKEN = 'https://oauth2.googleapis.com/token'
KEYS = 'https://www.googleapis.com/oauth2/v3/certs'


class GoogleOIDC:
    def __init__(self, settings):
        self.settings = settings
        self.keys = None
        self.keys_at = 0.0

    def client(self):
        return AsyncOAuth2Client(self.settings.google_client_id, self.settings.google_client_secret,
            scope='openid email', redirect_uri=self.settings.origin + '/auth/callback',
            code_challenge_method='S256', token_endpoint_auth_method='client_secret_post', timeout=5)

    async def authorize_url(self, state, nonce, verifier):
        async with self.client() as client:
            url, _ = client.create_authorization_url(AUTHORIZE, state=state, nonce=nonce, code_verifier=verifier)
            return url

    async def exchange(self, code, nonce, verifier):
        async with self.client() as client:
            token = await client.fetch_token(TOKEN, code=code, code_verifier=verifier)
        if self.keys is None or time.monotonic() - self.keys_at > 3600:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(KEYS)
                response.raise_for_status()
                self.keys = response.json()
                self.keys_at = time.monotonic()
        claims = JsonWebToken(['RS256']).decode(token['id_token'], self.keys,
            claims_cls=CodeIDToken,
            claims_options={'iss': {'essential': True, 'values': ['https://accounts.google.com', 'accounts.google.com']},
                            'aud': {'essential': True, 'value': self.settings.google_client_id},
                            'exp': {'essential': True}, 'iat': {'essential': True}, 'sub': {'essential': True}},
            claims_params={'nonce': nonce, 'client_id': self.settings.google_client_id,
                           'access_token': token.get('access_token')})
        claims.validate(leeway=30)
        if claims.get('email_verified') is not True or not isinstance(claims.get('email'), str):
            raise ValueError('Verified email required')
        if not claims.get('sub') or len(claims['sub']) > 255 or len(claims['email']) > 320:
            raise ValueError('Invalid identity')
        return claims['sub'], claims['email']
