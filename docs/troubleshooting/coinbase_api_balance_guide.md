# Coinbase API Troubleshooting: Zero Balance Issues

This guide consolidates current information from Coinbase documentation regarding API keys, permissions, and endpoints, specifically addressing the issue where an API key appears valid but returns a zero balance.

## 1. Determine Your API Type
Coinbase currently has three distinct API ecosystems. Using a key from one system on the endpoints of another is a common source of errors or empty data.

| API Type | Key Characteristics | Typical Endpoints |
| :--- | :--- | :--- |
| **Advanced Trade API** (Recommended) | Uses "Cloud API Keys" (CDP). Scoped to specific portfolios. | `api.coinbase.com/api/v3/brokerage/...` |
| **Legacy API** (v2) | Older keys. Often used for simple wallet operations. Being deprecated. | `api.coinbase.com/v2/accounts` |
| **International Exchange (INTX)** | Separate platform for non-US derivatives/spot. Distinct keys. | `api.international.coinbase.com/api/v1/...` |

> **Critical Check:** Ensure your library or code is targeting the endpoint matching your key type. You cannot use a Legacy v2 key to query Advanced Trade v3 endpoints, and vice versa.

## 2. The "Portfolio" Trap (Advanced Trade)
In the Advanced Trade (v3) architecture, funds are held in **Portfolios**.
*   **Default Portfolio:** Most users have a "Default" portfolio.
*   **Multiple Portfolios:** You can create additional portfolios.
*   **The Issue:** An API key is often **scoped to a specific portfolio** upon creation. If your funds are in the "Default" portfolio, but your key was created for (or defaults to) a different empty portfolio, you will see a valid response with zero balance.

**Solution:**
1.  **List Portfolios:** Use the `GET /api/v3/brokerage/portfolios` endpoint to see all portfolios and their UUIDs.
2.  **Check Key Scope:** Verify in the Coinbase Developer Portal if your key is restricted to a specific portfolio.
3.  **Specify Portfolio:** When querying balances (e.g., `GET /api/v3/brokerage/accounts`), you may need to explicitly pass the `portfolio_uuid` query parameter if your key has access to multiple.

## 3. Permissions & Scopes
Even if the key works, it might lack the specific permission to *read* balances.

*   **Legacy Keys:** Ensure the `wallet:accounts:read` permission is checked.
*   **Advanced Trade Keys:** Ensure "View" permissions are granted for "Accounts" and "Portfolios".
*   **INTX Keys:** Requires "View" permission specifically for the International Exchange section.

## 4. International Exchange (INTX) Specifics
If you are trading on the International Exchange (perpetuals/futures), your funds are in a completely separate "Trading Balance" from your standard Coinbase.com Spot wallet.

*   **Endpoint:** You must use `GET /api/v1/portfolios/{portfolio_id}/balances` (or `/intx/balances` depending on the client/version).
*   **Transfer:** Funds must be explicitly transferred from the "Spot" wallet to the "International" wallet to be visible here.
*   **Asset Type:** INTX often uses USDC for collateral. Ensure you are looking for the correct asset ID (e.g., `USDC` vs `USD`).

## 5. Common "Zero Balance" Gotchas

### A. Fiat vs. Crypto
*   Some endpoints separate "Accounts" (Crypto) from "Fiat Accounts" (USD/EUR).
*   Ensure you are not filtering for `BTC` when your funds are in `USD`.

### B. Pagination
*   If you have many dust balances (tiny amounts of old coins), your main active wallet might be on "Page 2" of the API response.
*   **Check:** Does your code handle `cursor` or `pagination` tokens? If not, you might only be seeing the first 25 (empty) accounts.

### C. UUID Mismatch
*   **Account UUID** != **Portfolio UUID** != **Wallet UUID**.
*   Ensure you are not passing a Portfolio UUID into an endpoint expecting an Account UUID.

## 6. Verification Checklist

1.  [ ] **Identify Key Type:** Is it a "CDP API Key" (Advanced Trade) or "Legacy"?
2.  [ ] **Verify Permissions:** Does it have `view` / `read` access?
3.  [ ] **Check Portfolio:** Call `GET /api/v3/brokerage/portfolios` (if v3). Do you see the portfolio with the funds?
4.  [ ] **Check Pagination:** Print the raw full JSON response to see if there is a `has_next: true` or similar field.
5.  [ ] **Test with CLI:** If possible, use the Coinbase CLI or a simple `curl` command to isolate the issue from your application code.
