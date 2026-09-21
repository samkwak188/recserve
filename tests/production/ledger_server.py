"""Isolated HTTPS S3 transport fixture; not a provider authorization/durability test."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import os
import ssl
from urllib.parse import urlsplit, unquote
from xml.sax.saxutils import escape

OBJECTS = {}


def chunks(stream):
    result = bytearray()
    while True:
        line = stream.readline(128)
        size = int(line.strip().split(b';')[0], 16)
        if size == 0:
            while stream.readline(1024) not in (b'\r\n', b'\n', b''):
                pass
            return bytes(result)
        if size > 8192 or len(result) + size > 8192:
            raise ValueError('Fixture body too large')
        block = stream.read(size)
        if len(block) != size or stream.read(2) != b'\r\n':
            raise ValueError('Incomplete fixture body')
        result.extend(block)


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'  # Includes bounded Expect: 100-continue handling.

    def log_message(self, *args):
        pass  # Never log credentials or encrypted preference/privacy content.

    def answer(self, status, body=b'', content_type='application/octet-stream'):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def allowed(self):
        return self.headers.get('Authorization', '').startswith('AWS4-HMAC-SHA256 ')

    def do_GET(self):
        if not self.allowed():
            return self.answer(403)
        path = unquote(urlsplit(self.path).path)
        if path.rstrip('/') == '/test-ledger':
            entries = ''.join(f'<Contents><Key>{escape(k)}</Key><Size>{len(v)}</Size></Contents>'
                              for k, v in sorted(OBJECTS.items()))
            body = ('<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                    '<Name>test-ledger</Name><IsTruncated>false</IsTruncated>' + entries + '</ListBucketResult>')
            return self.answer(200, body.encode(), 'application/xml')
        key = path.removeprefix('/test-ledger/')
        return self.answer(200, OBJECTS[key]) if key in OBJECTS else self.answer(404)

    def do_PUT(self):
        if not self.allowed() or not self.path.startswith('/test-ledger/deletions/'):
            return self.answer(403)
        try:
            if self.headers.get('Transfer-Encoding', '').lower() == 'chunked':
                body = chunks(self.rfile)
            else:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 8192:
                    return self.answer(413)
                body = self.rfile.read(size)
            if self.headers.get('Content-Encoding') == 'aws-chunked':
                body = chunks(io.BytesIO(body))
            if not body.startswith(b'gAAAA') or len(body) > 4096:
                return self.answer(400)
            OBJECTS[unquote(urlsplit(self.path).path).removeprefix('/test-ledger/')] = body
            self.answer(200)
        except (ValueError, OSError):
            self.answer(400)


if __name__ == '__main__':
    server = ThreadingHTTPServer(('0.0.0.0', 5000), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain('/fixture/cert.pem', '/fixture/key.pem')
    server.socket = context.wrap_socket(server.socket, server_side=True)
    server.serve_forever()
