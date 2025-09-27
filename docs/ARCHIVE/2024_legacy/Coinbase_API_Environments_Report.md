---
status: deprecated
archived: 2024-12-31
reason: Pre-perpetuals documentation from Alpaca/equities era
---

# ⚠️ DEPRECATED DOCUMENT

This document is from the legacy Alpaca/Equities version of GPT-Trader and is no longer current.
The project has migrated to Coinbase Perpetual Futures.

For current documentation, see: [docs/README.md](/docs/README.md)

---


**To:** Project Manager
**From:** Gemini AI Assistant
**Date:** 2025-08-31
**Subject:** Report on Coinbase API Environments: Sandbox vs. Production Setup

### 1. Executive Summary

This report details the configuration of our project for Coinbase Sandbox and Production environments. Our primary goal is to ensure readiness for live perpetual futures trading.

The investigation confirms that the **Sandbox environment does not support perpetual futures trading**. The Sandbox is limited to the legacy Exchange API, while perpetuals are only available through the modern Advanced Trade API in the Production environment.

Our current system supports switching between these environments via the `COINBASE_SANDBOX` environment variable. However, there is a critical configuration gap: the system does not account for the separate API keys required for Sandbox and Production, leading to a cumbersome and error-prone setup.

This report recommends specific configuration and code changes to properly support both environments and clarifies the limited, but still useful, role of the Sandbox for testing non-perpetuals features.

### 2. Key Differences: Sandbox vs. Production

Based on official documentation and an analysis of our codebase, the following are the key differences:

| Feature | Production Environment | Sandbox Environment |
| :--- | :--- | :--- |
| **Primary Use** | Live trading with real funds. | Testing API integration with mocked data. |
| **API Endpoint** | `https://api.coinbase.com` | `https://api-public.sandbox.exchange.coinbase.com` |
| **API Keys** | Requires unique Production API keys. | Requires **separate** API keys generated from the Coinbase Sandbox. |
| **API Mode** | Advanced Trade API (supports perpetuals). | Legacy Exchange API (no perpetuals support). |
| **Perpetual Futures**| **Available** | **Not Available** |

### 3. Current Project Configuration Analysis

Our codebase is designed to switch between environments, but has a significant flaw.

- **Environment Switch:** The `src/bot_v2/orchestration/broker_factory.py` file correctly uses the `COINBASE_SANDBOX` environment variable to select the appropriate API base URL. When `COINBASE_SANDBOX=1`, it correctly points to the sandbox URL.
- **Forced API Mode:** The factory correctly forces the `api_mode` to `"exchange"` when in Sandbox mode, acknowledging the sandbox's limitations.
- **Configuration Gap:** The system reads the same environment variables (`COINBASE_API_KEY`, `COINBASE_API_SECRET`) for both environments. The official documentation requires separate keys for each. This means a developer must manually edit the `.env` file to switch between Sandbox and Production, which is inefficient and increases the risk of accidentally using live keys in a test setting.

### 4. Recommendations

To ensure a robust and safe development workflow, the following actions are recommended:

1.  **Update Environment Variable Configuration:**
    *   Modify `.env.template` to include separate variables for sandbox credentials.
    *   **Recommendation:**
        ```
        # Production Keys
        COINBASE_PROD_API_KEY=your-production-api-key
        COINBASE_PROD_API_SECRET=your-production-api-secret

        # Sandbox Keys
        COINBASE_SANDBOX_API_KEY=your-sandbox-api-key
        COINBASE_SANDBOX_API_SECRET=your-sandbox-api-secret
        ```

2.  **Refactor the Broker Factory:**
    *   Update the `create_brokerage` function in `src/bot_v2/orchestration/broker_factory.py`.
    *   The function should read the appropriate set of API keys based on the value of the `COINBASE_SANDBOX` flag.

3.  **Define a Clear Testing Strategy:**
    *   Acknowledge that all development and testing related to **perpetual futures must be done in the Production environment**, ideally with a dedicated, low-funded test portfolio.
    *   Utilize the **Sandbox environment** for testing other core functionalities, such as:
        *   Spot market order placement and cancellation logic.
        *   API connectivity and authentication logic.
        *   Error handling and response parsing.
        *   Basic account data retrieval.

By implementing these changes, we can create a more secure and efficient development process, clearly delineating what can be tested in the sandbox versus what must be validated in the live environment. This minimizes risk while preparing for our primary goal of deploying the live perpetual futures trading bot.
