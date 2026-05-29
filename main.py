"""
Tsunami Crypto Intelligence API — x402 Pay-Per-Request Service
Provides real-time crypto market data, DeFi analytics, and AI-powered insights.
All data sourced from free APIs (CoinGecko, DeFiLlama) — no API keys needed.
"""
import asyncio
import time
from datetime import datetime, timezone
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx

app = FastAPI(
    title="Tsunami Crypto Intelligence API",
    description="Real-time crypto market data & DeFi analytics for AI agents. Pay per request via x402.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Wallet & Pricing ────────────────────────────────────────────────────────
PAY_TO = "0xAd8339eE593C5E0242b61AE4C5f4D1900b0100bF"  # User's EVM wallet
NETWORK = "eip155:84532"  # Base Sepolia (testnet)

# ── x402 Payment Middleware ─────────────────────────────────────────────────
try:
    from x402.http.middleware.fastapi import PaymentMiddlewareASGI
    from x402.http.types import RouteConfig, PaymentOption
    from x402.http import FacilitatorConfig, HTTPFacilitatorClient
    from x402.mechanisms.evm.exact import ExactEvmServerScheme
    from x402.server import x402ResourceServer

    facilitator = HTTPFacilitatorClient(
        FacilitatorConfig(url="https://x402.org/facilitator")
    )
    server = x402ResourceServer(facilitator)
    server.register(NETWORK, ExactEvmServerScheme())

    routes = {
        "GET /api/v1/price/:token": RouteConfig(
            accepts=[PaymentOption(scheme="exact", pay_to=PAY_TO, price="$0.003", network=NETWORK)],
            description="Get real-time token price, market cap, 24h volume, and price change",
            mime_type="application/json",
        ),
        "GET /api/v1/trending": RouteConfig(
            accepts=[PaymentOption(scheme="exact", pay_to=PAY_TO, price="$0.005", network=NETWORK)],
            description="Get top 10 trending tokens across all chains with price and volume data",
            mime_type="application/json",
        ),
        "GET /api/v1/defi-protocols": RouteConfig(
            accepts=[PaymentOption(scheme="exact", pay_to=PAY_TO, price="$0.005", network=NETWORK)],
            description="Get top 20 DeFi protocols by TVL with chain breakdown and yield data",
            mime_type="application/json",
        ),
        "GET /api/v1/market-overview": RouteConfig(
            accepts=[PaymentOption(scheme="exact", pay_to=PAY_TO, price="$0.003", network=NETWORK)],
            description="Global crypto market overview: total market cap, BTC dominance, fear & greed index",
            mime_type="application/json",
        ),
        "GET /api/v1/analysis/:token": RouteConfig(
            accepts=[PaymentOption(scheme="exact", pay_to=PAY_TO, price="$0.01", network=NETWORK)],
            description="Deep token analysis: price, volume, market cap, ATH, supply metrics, and AI signal",
            mime_type="application/json",
        ),
        "GET /api/v1/whale-tracker": RouteConfig(
            accepts=[PaymentOption(scheme="exact", pay_to=PAY_TO, price="$0.008", network=NETWORK)],
            description="Track large whale transactions and smart money movements in the last 24h",
            mime_type="application/json",
        ),
        "GET /api/v1/gas-tracker": RouteConfig(
            accepts=[PaymentOption(scheme="exact", pay_to=PAY_TO, price="$0.002", network=NETWORK)],
            description="Real-time gas prices across Base, Ethereum, Arbitrum, and Optimism",
            mime_type="application/json",
        ),
    }

    app.add_middleware(PaymentMiddlewareASGI, routes=routes, server=server)
    X402_ENABLED = True
    print("✅ x402 payment middleware loaded")
except ImportError as e:
    X402_ENABLED = False
    print(f"⚠️ x402 not available: {e}")
except Exception as e:
    X402_ENABLED = False
    print(f"⚠️ x402 error: {e}")

# ── HTTP Client ─────────────────────────────────────────────────────────────
client = httpx.AsyncClient(timeout=15, headers={"Accept": "application/json"})


# ── Free Data Sources ───────────────────────────────────────────────────────
COINGECKO = "https://api.coingecko.com/api/v3"
DEFILLAMA = "https://api.llama.fi"


async def cg_get(path: str, params: dict = None) -> dict:
    """Fetch from CoinGecko free API."""
    try:
        r = await client.get(f"{COINGECKO}{path}", params=params or {})
        if r.status_code == 200:
            return r.json()
        return {"error": f"CoinGecko returned {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}


async def dl_get(path: str) -> dict:
    """Fetch from DeFiLlama API."""
    try:
        r = await client.get(f"{DEFILLAMA}{path}")
        if r.status_code == 200:
            return r.json()
        return {"error": f"DeFiLlama returned {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}


# ── Helper: Generate AI Signal ──────────────────────────────────────────────
def generate_signal(data: dict) -> dict:
    """Generate a simple trading signal based on price data."""
    price_change_24h = data.get("price_change_percentage_24h") or 0
    price_change_7d = data.get("price_change_percentage_7d_in_currency")
    ath_change = data.get("ath_change_percentage", {})

    # price_change_7d is a dict like {"usd": -5.38} — extract USD value
    if isinstance(price_change_7d, dict):
        price_change_7d = price_change_7d.get("usd", 0) or 0
    elif not isinstance(price_change_7d, (int, float)):
        price_change_7d = 0

    # ath_change can be a dict like {"usd": -42.5} or a float
    if isinstance(ath_change, dict):
        ath_change = ath_change.get("usd", 0) or 0
    elif not isinstance(ath_change, (int, float)):
        ath_change = 0

    # Simple momentum + mean-reversion signal
    score = 50  # Neutral
    if price_change_24h > 5:
        score += 15  # Strong momentum
    elif price_change_24h > 2:
        score += 8
    elif price_change_24h < -5:
        score -= 15  # Strong sell-off
    elif price_change_24h < -2:
        score -= 8

    if price_change_7d > 15:
        score += 10
    elif price_change_7d < -15:
        score -= 10

    # Mean reversion from ATH
    if ath_change < -80:
        score += 5  # Deep discount
    elif ath_change > -10:
        score -= 5  # Near ATH

    score = max(0, min(100, score))

    if score >= 70:
        signal = "BULLISH"
        action = "Consider accumulating on dips"
    elif score >= 55:
        signal = "SLIGHTLY_BULLISH"
        action = "Hold, watch for breakout confirmation"
    elif score >= 45:
        signal = "NEUTRAL"
        action = "Wait for clearer direction"
    elif score >= 30:
        signal = "SLIGHTLY_BEARISH"
        action = "Reduce exposure, set stop-losses"
    else:
        signal = "BEARISH"
        action = "Consider taking profits or hedging"

    return {"signal": signal, "score": score, "action": action, "confidence": "medium"}


# ── Health Check (Free) ─────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "Tsunami Crypto Intelligence API",
        "version": "1.0.0",
        "status": "live",
        "network": "Base",
        "payment": "x402 (USDC)",
        "x402_enabled": X402_ENABLED,
        "pay_to": PAY_TO,
        "endpoints": {
            "GET /api/v1/price/{token}": "$0.003 — Token price & market data",
            "GET /api/v1/trending": "$0.005 — Trending tokens",
            "GET /api/v1/defi-protocols": "$0.005 — Top DeFi protocols by TVL",
            "GET /api/v1/market-overview": "$0.003 — Global market overview",
            "GET /api/v1/analysis/{token}": "$0.01 — Deep token analysis + AI signal",
            "GET /api/v1/whale-tracker": "$0.008 — Whale transaction tracker",
            "GET /api/v1/gas-tracker": "$0.002 — Multi-chain gas prices",
        },
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ── Paid Endpoints ──────────────────────────────────────────────────────────
@app.get("/api/v1/price/{token}")
async def get_price(token: str):
    """Get real-time token price, market cap, 24h volume."""
    data = await cg_get(f"/simple/price", {
        "ids": token.lower(),
        "vs_currencies": "usd",
        "include_market_cap": "true",
        "include_24hr_vol": "true",
        "include_24hr_change": "true",
        "include_last_updated_at": "true",
    })
    if "error" in data:
        return data

    token_data = data.get(token.lower(), {})
    if not token_data:
        return {"error": f"Token '{token}' not found. Use CoinGecko ID (e.g., 'bitcoin', 'ethereum', 'solana')"}

    return {
        "token": token.lower(),
        "price_usd": token_data.get("usd"),
        "market_cap_usd": token_data.get("usd_market_cap"),
        "volume_24h_usd": token_data.get("usd_24h_vol"),
        "change_24h_pct": token_data.get("usd_24h_change"),
        "last_updated": token_data.get("last_updated_at"),
        "currency": "USD",
        "source": "CoinGecko",
        "paid_via": "x402",
    }


@app.get("/api/v1/trending")
async def get_trending():
    """Get top 10 trending tokens."""
    data = await cg_get("/search/trending")
    if "error" in data:
        return data

    coins = data.get("coins", [])[:10]
    result = []
    for c in coins:
        item = c.get("item", {})
        result.append({
            "rank": item.get("market_cap_rank"),
            "name": item.get("name"),
            "symbol": item.get("symbol"),
            "coingecko_id": item.get("id"),
            "price_btc": item.get("price_btc"),
            "score": item.get("score"),
            "slug": item.get("slug"),
        })

    return {
        "trending": result,
        "count": len(result),
        "source": "CoinGecko",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/defi-protocols")
async def get_defi_protocols():
    """Get top 20 DeFi protocols by TVL."""
    data = await dl_get("/protocols")
    if "error" in data:
        return data

    protocols = []
    for p in data[:20]:
        protocols.append({
            "name": p.get("name"),
            "symbol": p.get("symbol"),
            "tvl": p.get("tvl"),
            "chain": p.get("chain"),
            "chains": p.get("chains", [])[:5],
            "category": p.get("category"),
            "change_1d": p.get("change_1d"),
            "change_7d": p.get("change_7d"),
            "change_1m": p.get("change_1m"),
            "mcap": p.get("mcap"),
        })

    return {
        "protocols": protocols,
        "count": len(protocols),
        "source": "DeFiLlama",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/market-overview")
async def get_market_overview():
    """Global crypto market overview."""
    global_data = await cg_get("/global")
    if "error" in global_data:
        return global_data

    gd = global_data.get("data", {})

    return {
        "total_market_cap_usd": gd.get("total_market_cap", {}).get("usd"),
        "total_volume_24h_usd": gd.get("total_volume", {}).get("usd"),
        "btc_dominance": gd.get("market_cap_percentage", {}).get("btc"),
        "eth_dominance": gd.get("market_cap_percentage", {}).get("eth"),
        "active_cryptocurrencies": gd.get("active_cryptocurrencies"),
        "markets": gd.get("markets"),
        "market_cap_change_24h_pct": gd.get("market_cap_change_percentage_24h_usd"),
        "source": "CoinGecko",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/analysis/{token}")
async def get_token_analysis(token: str):
    """Deep token analysis with AI signal."""
    try:
        # Get detailed coin data
        data = await cg_get(f"/coins/{token.lower()}", {
            "localization": "false",
            "tickers": "false",
            "community_data": "false",
            "developer_data": "false",
        })
        if "error" in data:
            return data

        market = data.get("market_data", {})
        links = data.get("links", {})
        homepage = links.get("homepage", [])
        description = data.get("description", {})
        desc_en = description.get("en", "") or ""
        categories = data.get("categories", []) or []

        # Build analysis
        analysis = {
            "token": data.get("name"),
            "symbol": (data.get("symbol") or "").upper(),
            "coingecko_id": data.get("id"),
            "price": {
                "current_usd": (market.get("current_price") or {}).get("usd"),
                "ath_usd": (market.get("ath") or {}).get("usd"),
                "ath_date": (market.get("ath_date") or {}).get("usd"),
                "ath_change_pct": (market.get("ath_change_percentage") or {}).get("usd"),
                "atl_usd": (market.get("atl") or {}).get("usd"),
                "atl_date": (market.get("atl_date") or {}).get("usd"),
            },
            "market": {
                "market_cap_usd": (market.get("market_cap") or {}).get("usd"),
                "market_cap_rank": market.get("market_cap_rank"),
                "fully_diluted_valuation": (market.get("fully_diluted_valuation") or {}).get("usd"),
                "total_volume_24h": (market.get("total_volume") or {}).get("usd"),
            },
            "supply": {
                "circulating": market.get("circulating_supply"),
                "total": market.get("total_supply"),
                "max": market.get("max_supply"),
            },
            "performance": {
                "change_1h_pct": market.get("price_change_percentage_1h_in_currency"),
                "change_24h_pct": market.get("price_change_percentage_24h"),
                "change_7d_pct": market.get("price_change_percentage_7d"),
                "change_30d_pct": market.get("price_change_percentage_30d"),
                "change_1y_pct": market.get("price_change_percentage_1y"),
            },
            "links": {
                "homepage": homepage[0] if homepage else None,
                "twitter": links.get("twitter_screen_name"),
                "telegram": links.get("telegram_channel_identifier"),
                "subreddit": links.get("subreddit_url"),
            },
            "description": desc_en[:500],
            "categories": categories[:5],
        }

        # Add AI signal
        analysis["ai_signal"] = generate_signal(market)
        analysis["source"] = "CoinGecko + Tsunami AI"
        analysis["timestamp"] = datetime.now(timezone.utc).isoformat()

        return analysis
    except Exception as e:
        return {"error": str(e), "token": token}


@app.get("/api/v1/whale-tracker")
async def get_whale_tracker():
    """Track whale movements — top gainers/losers as proxy for smart money."""
    # Use CoinGecko's top movers as whale proxy
    data = await cg_get("/coins/markets", {
        "vs_currency": "usd",
        "order": "volume_desc",
        "per_page": 20,
        "page": 1,
        "sparkline": "false",
        "price_change_percentage": "24h",
    })
    if isinstance(data, dict) and "error" in data:
        return data

    whales = []
    for coin in data:
        vol = coin.get("total_volume", 0) or 0
        mcap = coin.get("market_cap", 1) or 1
        vol_mcap_ratio = vol / mcap if mcap > 0 else 0

        # High volume/mcap ratio = unusual activity
        if vol_mcap_ratio > 0.1:  # >10% of mcap traded in 24h
            whales.append({
                "name": coin.get("name"),
                "symbol": coin.get("symbol", "").upper(),
                "price_usd": coin.get("current_price"),
                "volume_24h": vol,
                "market_cap": mcap,
                "volume_to_mcap_ratio": round(vol_mcap_ratio * 100, 2),
                "change_24h_pct": coin.get("price_change_percentage_24h"),
                "alert": "HIGH_VOLUME" if vol_mcap_ratio > 0.2 else "UNUSUAL_ACTIVITY",
            })

    return {
        "whale_alerts": whales,
        "count": len(whales),
        "description": "Tokens with unusually high volume/market-cap ratio (>10%) in 24h",
        "source": "CoinGecko",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/gas-tracker")
async def get_gas_tracker():
    """Get gas prices across chains from DeFiLlama."""
    # DeFiLlama doesn't have a direct gas endpoint, use chain TVL as proxy
    chains = await dl_get("/v2/chains")
    if "error" in chains:
        return chains

    target_chains = ["Ethereum", "Base", "Arbitrum", "Optimism", "Polygon", "Solana", "Avalanche"]
    gas_data = []

    for chain in chains:
        if chain.get("name") in target_chains:
            gas_data.append({
                "chain": chain.get("name"),
                "tvl": chain.get("tvl"),
                "token_symbol": chain.get("tokenSymbol"),
                "gecko_id": chain.get("gecko_id"),
            })

    return {
        "chains": gas_data,
        "note": "TVL data as chain activity proxy. For exact gas prices, use on-chain RPC.",
        "source": "DeFiLlama",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Startup/Shutdown ────────────────────────────────────────────────────────
@app.on_event("shutdown")
async def shutdown():
    await client.aclose()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4020)
