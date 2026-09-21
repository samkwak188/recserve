import time
import httpx
import pytest
import respx
from authlib.jose import JsonWebKey, JsonWebToken
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption
from recserve_app.oidc import GoogleOIDC, TOKEN, KEYS


@pytest.fixture(scope='module')
def keys():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    public = JsonWebKey.import_key(private.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo),
                                  {'kid': 'fixture', 'use': 'sig'}).as_dict()
    return pem, public


@pytest.mark.asyncio
@pytest.mark.parametrize('changed', [None, {'iss': 'https://attacker.invalid'}, {'aud': 'other-client'},
    {'exp': 1}, {'nonce': 'wrong'}, {'email_verified': False}, {'email_verified': 'true'}, {'sub': ''}])
async def test_signed_claims(app, keys, changed):
    private, public = keys
    stamp = int(time.time())
    claims = dict(iss='https://accounts.google.com', aud='test-client', sub='google-subject',
                  iat=stamp, exp=stamp + 300, nonce='expected', email='one@example.invalid', email_verified=True)
    if changed:
        claims.update(changed)
    token = JsonWebToken(['RS256']).encode({'alg': 'RS256', 'kid': 'fixture'}, claims, private).decode()
    with respx.mock as mock:
        endpoint = mock.post(TOKEN).mock(return_value=httpx.Response(200, json={
            'access_token': 'fixture-access', 'token_type': 'Bearer', 'id_token': token}))
        mock.get(KEYS).mock(return_value=httpx.Response(200, json={'keys': [public]}))
        oidc = GoogleOIDC(app.state.settings)
        if changed:
            with pytest.raises(Exception):
                await oidc.exchange('auth-code', 'expected', 'proof-key')
        else:
            assert await oidc.exchange('auth-code', 'expected', 'proof-key') == ('google-subject', 'one@example.invalid')
            assert b'code_verifier=proof-key' in endpoint.calls[0].request.content


@pytest.mark.asyncio
async def test_forged_signature(app, keys):
    private, public = keys
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
        Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    token = JsonWebToken(['RS256']).encode({'alg': 'RS256', 'kid': 'fixture'}, dict(
        iss='https://accounts.google.com', aud='test-client', sub='s', iat=int(time.time()),
        exp=int(time.time()) + 300, nonce='n', email='one@example.invalid', email_verified=True), other).decode()
    with respx.mock as mock:
        mock.post(TOKEN).mock(return_value=httpx.Response(200, json={
            'access_token': 'x', 'token_type': 'Bearer', 'id_token': token}))
        mock.get(KEYS).mock(return_value=httpx.Response(200, json={'keys': [public]}))
        with pytest.raises(Exception):
            await GoogleOIDC(app.state.settings).exchange('code', 'n', 'verifier')
