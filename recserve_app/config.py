from dataclasses import dataclass
import os
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Settings:
    database_url: str
    origin: str
    google_client_id: str
    google_client_secret: str
    bundle: str = ''
    upstream_host: str = '127.0.0.1'
    upstream_port: int = 9400
    consent_version: str = 'research-v1'

    def __post_init__(self):
        parsed = urlsplit(self.origin)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.path or parsed.query or parsed.fragment or parsed.username:
            raise ValueError('APP_ORIGIN must be an HTTPS origin without a path')
        if not self.database_url.startswith('postgresql+psycopg://'):
            raise ValueError('PostgreSQL with psycopg is required')
        if not self.google_client_id or not self.google_client_secret:
            raise ValueError('Google OIDC credentials are required')

    @classmethod
    def from_env(cls):
        return cls(database_url=os.environ['DATABASE_URL'], origin=os.environ['APP_ORIGIN'],
                   google_client_id=os.environ['GOOGLE_CLIENT_ID'],
                   google_client_secret=os.environ['GOOGLE_CLIENT_SECRET'],
                   bundle=os.environ.get('MODEL_BUNDLE', ''),
                   upstream_host=os.environ.get('RETRIEVAL_HOST', '127.0.0.1'),
                   upstream_port=int(os.environ.get('RETRIEVAL_PORT', '9400')))
