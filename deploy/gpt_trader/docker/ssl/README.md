# TLS Certificates (Nginx)

The `production` compose profile mounts this directory into the nginx container
at `/etc/nginx/ssl` and expects the following files:

- `server.crt`
- `server.key`

For local testing, you can generate a self-signed certificate:

```bash
cd deploy/gpt_trader/docker
mkdir -p ssl
openssl req -x509 -newkey rsa:4096 -sha256 -days 3650 -nodes \
  -keyout ssl/server.key -out ssl/server.crt \
  -subj "/CN=localhost"
```

Note: do not commit real certificates or private keys to the repository.
