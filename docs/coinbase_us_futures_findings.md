# Coinbase US Futures Availability & API Access Findings

## Summary of Findings (November 23, 2025)

We investigated the availability of Perpetual Futures for US-based clients via the Coinbase Advanced Trade API to troubleshoot a `ProductID "BTC-PERP" could not be found` error during a dry-run.

### Key Discoveries

1.  **Product Availability:** 
    *   Coinbase *does* offer perpetual-style futures for US clients (launched July 2025) through **Coinbase Financial Markets (CFM)**.
    *   These products are legally distinct from the international "Perpetual Futures" and are technically "long-dated" futures (e.g., 5-year expiry) to comply with CFTC regulations.

2.  **Product IDs:**
    *   **International Exchange:** Uses IDs like `BTC-PERP`.
    *   **US (CFM) Futures:** Uses different IDs. 
        *   **Nano Bitcoin Perp-Style:** `BIP` (or potentially `BIT` for standard futures).
        *   **Nano Ether Perp-Style:** `ETP`.
    *   *The standard `BTC-PERP` ID is NOT visible to US spot-only API keys.*

3.  **Access Requirements:**
    *   **Separate Approval:** Access requires a specific application and approval for a Futures account with Coinbase Financial Markets.
    *   **API Key Permissions:** Standard "Trade" permissions on a spot account do **not** automatically grant access to these futures products. The API key must be associated with the futures-enabled account.

### Impact on Project Road Map

*   **Immediate Testing (Dry-Run):** 
    *   We cannot use `BTC-PERP` for the initial dry-run because the current API key is restricted to Spot trading.
    *   **Decision:** We will fallback to **Spot Trading (`BTC-USD`)** for the initial `canary` profile dry-run to verify the bot's infrastructure, connectivity, and logic.

*   **Future Development:**
    *   To support US Futures, we will need to:
        1.  Update the `ProductCatalog` or configuration to support `BIP` / `ETP` product IDs.
        2.  Document the requirement for users to apply for CFM Futures access.
        3.   potentially handle "Nano" contract sizing logic (1/100th BTC) if it differs from standard perp sizing.

## Reference
*   **US Perp Product ID:** `BIP` (Nano Bitcoin Perp-Style)
*   **International Perp Product ID:** `BTC-PERP`
