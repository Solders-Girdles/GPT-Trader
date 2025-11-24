# Kubernetes Deployment Notes

1. Copy the secret template and populate real credentials:
   ```bash
   cp deploy/gpt_trader/kubernetes/secrets/trading-bot-secrets.env.example \
      deploy/gpt_trader/kubernetes/secrets/trading-bot-secrets.env
   ```
2. Render and apply with Kustomize:
   ```bash
   kubectl apply -k deploy/gpt_trader/kubernetes
   ```
3. The deployment runs as a non-root user with a read-only root filesystem and no Linux capabilities.
   Mount `logs`, `data`, and `cache` volumes if additional writable paths are required.
4. Keep sample values out of overlays—`trading-bot-secrets.env` is git-ignored on purpose.
