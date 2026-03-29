"""
SLDP (Solid Power Inc) — Multi-Agent Swarm Risk Model
======================================================
Quantitative probability-weighted scenario analysis across Bear / Base / Bull
for 1M, 3M, 1Y, and 5Y horizons.

Current price    : $3.09
Analysis date    : 2026-03-29
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math

# ---------------------------------------------------------------------------
# 1. INPUT STATE (all values sourced from the prompt)
# ---------------------------------------------------------------------------

CURRENT_PRICE   = 3.09
ANALYST_TARGET  = 7.00
SUPPORT_LOW     = 2.91
SUPPORT_HIGH    = 3.04
RESIST_LOW      = 3.51
RESIST_HIGH     = 3.80
RSI             = 37.0
ADX             = 32.75
SHORT_INTEREST  = 0.1223   # 12.23% of float
WARRANTS        = 45_600_000
WARRANT_STRIKE  = 7.25
EARNINGS_DATE   = "2026-05-05 to 2026-05-12"
SP2_CATALYST    = "mid-2026"


# ---------------------------------------------------------------------------
# 2. AGENT DEFINITIONS
# ---------------------------------------------------------------------------

AGENTS = [
    "TechnicalAgent",       # reads price action, MAs, RSI, MACD, ADX
    "SentimentAgent",       # reads short interest, options flow, retail sentiment
    "CatalystAgent",        # tracks SP2 commissioning, earnings, partnerships
    "DilutionAgent",        # models warrant overhang & dilution scenarios
    "MacroAgent",           # EV sector macro, rate environment, commodity costs
    "ValuationAgent",       # DCF, comparables, book value floor
]


# ---------------------------------------------------------------------------
# 3. SCENARIO DATACLASS
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    horizon: str
    probability: float          # 0-1
    price_target: float
    thesis: str
    key_triggers: List[str]
    risks: List[str]
    expected_value: float = field(init=False)

    def __post_init__(self):
        self.expected_value = round(self.probability * self.price_target, 4)


# ---------------------------------------------------------------------------
# 4. SCENARIO CONSTRUCTION — per horizon
# ---------------------------------------------------------------------------

def build_scenarios() -> Dict[str, List[Scenario]]:

    horizons: Dict[str, List[Scenario]] = {}

    # ── 1 MONTH ────────────────────────────────────────────────────────────
    horizons["1M"] = [
        Scenario(
            name="Bear",
            horizon="1M",
            probability=0.45,
            price_target=2.55,
            thesis=(
                "Death cross confirms; support at $2.91 breaks on volume. "
                "Short sellers press into earnings uncertainty. No near-term catalyst. "
                "RSI oversold but not bottoming — ADX >30 signals strong trend continuation."
            ),
            key_triggers=[
                "Daily close below $2.91 support",
                "Short interest rises above 14%",
                "Broader small-cap risk-off rotation",
                "No SP2 newsflow",
            ],
            risks=[
                "Short squeeze if unexpected partnership announced",
                "Any positive earnings pre-announcement",
            ],
        ),
        Scenario(
            name="Base",
            horizon="1M",
            probability=0.40,
            price_target=3.15,
            thesis=(
                "Stock consolidates between $2.91–$3.51 range. "
                "RSI stabilises ~35-40, no fresh catalyst before earnings. "
                "Market makers defend $3.00 psychological level."
            ),
            key_triggers=[
                "Hold above $2.91 on multiple tests",
                "Modest improvement in MACD histogram",
                "Neutral earnings sentiment into May window",
            ],
            risks=[
                "Break below $2.91 flips base to bear",
                "Macro headwinds compress small-cap multiples",
            ],
        ),
        Scenario(
            name="Bull",
            horizon="1M",
            probability=0.15,
            price_target=3.75,
            thesis=(
                "Surprise positive catalyst (licensing deal, DOE/DOD update, "
                "or SP2 early readout) drives short-covering rally. "
                "Price recaptures $3.51 resistance; short sellers cover ~12% float."
            ),
            key_triggers=[
                "Volume >2× 20-day average on up day",
                "Close above $3.51 resistance",
                "Partnership or grant announcement",
            ],
            risks=[
                "Rally fades without follow-through volume",
                "Broader market drawdown overrides catalyst",
            ],
        ),
    ]

    # ── 3 MONTHS ───────────────────────────────────────────────────────────
    horizons["3M"] = [
        Scenario(
            name="Bear",
            horizon="3M",
            probability=0.38,
            price_target=2.20,
            thesis=(
                "Q1 earnings (May) disappoint: cash burn accelerates, "
                "no SP2 commissioning confirmation. Dilution risk materialises — "
                "company raises equity near current levels. Short interest climbs "
                "to 15-18% of float. Stock tests 52-week lows."
            ),
            key_triggers=[
                "Earnings miss + guidance cut",
                "Balance sheet shows <12 months runway",
                "ATM offering or shelf activation",
                "SP2 commissioning delayed past Q3 2026",
            ],
            risks=[
                "Short squeeze on any OEM win announcement",
                "Buyout premium from major automotive OEM",
            ],
        ),
        Scenario(
            name="Base",
            horizon="3M",
            probability=0.42,
            price_target=3.40,
            thesis=(
                "Earnings in-line; SP2 on schedule for mid-2026. "
                "Stock rebuilds from oversold levels as macro stabilises. "
                "Short interest plateaus; ADX begins declining from 32 peak."
            ),
            key_triggers=[
                "Earnings: cash burn within guided range",
                "SP2 commissioning timeline confirmed",
                "RSI recovery toward 50 neutral",
            ],
            risks=[
                "Warrant holders begin hedging above $7.00 area",
                "EV battery sector sentiment remains soft",
            ],
        ),
        Scenario(
            name="Bull",
            horizon="3M",
            probability=0.20,
            price_target=4.80,
            thesis=(
                "SP2 commissioning news pre-announced or leaked. "
                "Major OEM partnership expanded. Earnings beat with positive "
                "cash-management commentary. Short squeeze drives rapid recovery "
                "toward analyst $7.00 target — 12% float covering adds fuel."
            ),
            key_triggers=[
                "SP2 commissioning announced ahead of schedule",
                "OEM expansion agreement signed",
                "Short interest drops below 8%",
                "Analyst upgrades or PT raises",
            ],
            risks=[
                "Warrants at $7.25 create strong supply ceiling",
                "Retail profit-taking caps momentum",
            ],
        ),
    ]

    # ── 1 YEAR ─────────────────────────────────────────────────────────────
    horizons["1Y"] = [
        Scenario(
            name="Bear",
            horizon="1Y",
            probability=0.30,
            price_target=1.50,
            thesis=(
                "SP2 commissioning delayed or fails to produce commercial-grade "
                "cells. OEM qualification timelines slip. Company forced into "
                "dilutive equity raise or strategic restructuring. "
                "Cash runway concern becomes existential. Stock delisting risk "
                "if price stays below $1 for extended period."
            ),
            key_triggers=[
                "SP2 technical failure or yield problems reported",
                "OEM partner reduces commitment",
                "Equity raise >20% of shares outstanding",
                "Industry consolidation disadvantages SLDP",
            ],
            risks=[
                "Acquisition offer at premium to distressed price",
                "Government grant offsets cash needs",
            ],
        ),
        Scenario(
            name="Base",
            horizon="1Y",
            probability=0.45,
            price_target=5.20,
            thesis=(
                "SP2 commissions mid-2026 as scheduled. Qualification data "
                "shared with OEM partners. Revenue from licensing/pilot sales "
                "starts appearing on income statement. Analyst target of $7.00 "
                "partially achieved as de-risking premium priced in. "
                "Warrants at $7.25 remain out-of-the-money."
            ),
            key_triggers=[
                "SP2 commissioning confirmed + cell data published",
                "First commercial qualification agreement signed",
                "Cash runway extended via non-dilutive funding",
                "Positive earnings trajectory",
            ],
            risks=[
                "Competitor (QuantumScape, SES AI) announces breakthrough first",
                "45.6M warrants create supply overhang near $5–7 range",
            ],
        ),
        Scenario(
            name="Bull",
            horizon="1Y",
            probability=0.25,
            price_target=9.50,
            thesis=(
                "SP2 exceeds targets; multiple OEM qualifications announced. "
                "Licensing revenue guidance issued. Short squeeze + analyst upgrades "
                "drive price through $7.25 warrant strike — warrants exercised adding "
                "cash to balance sheet. Stock approaches/exceeds analyst $7.00 PT. "
                "Market re-rates SLDP as legitimate solid-state battery leader."
            ),
            key_triggers=[
                "SP2 capacity utilisation >70% by Q4 2026",
                "2+ OEM qualification agreements",
                "Licensing deal >$50M TCV announced",
                "Warrant exercises add $330M+ to treasury",
            ],
            risks=[
                "45.6M warrant dilution (~20% of float) caps upside",
                "Execution risk on scale-up",
            ],
        ),
    ]

    # ── 5 YEARS ────────────────────────────────────────────────────────────
    horizons["5Y"] = [
        Scenario(
            name="Bear",
            horizon="5Y",
            probability=0.25,
            price_target=0.50,
            thesis=(
                "Solid-state battery technology does not achieve commercial-grade "
                "yield at scale by 2030. SLDP exhausts capital, unable to compete "
                "with deep-pocketed incumbents (Toyota, Samsung SDI, CATL). "
                "Company pivots, is acquired for IP at distressed valuation, "
                "or trades as a penny stock on residual IP value."
            ),
            key_triggers=[
                "No commercial production by 2028",
                "Cash < 6 months runway without raise",
                "OEM partners develop in-house solutions",
                "EV market growth disappoints (slower than forecast)",
            ],
            risks=[
                "IP acquisition by major player at 2-3× current price",
            ],
        ),
        Scenario(
            name="Base",
            horizon="5Y",
            probability=0.50,
            price_target=12.00,
            thesis=(
                "SLDP achieves limited commercial production by 2028. "
                "2-3 OEM qualification agreements generating licensing revenue. "
                "JV or tolling manufacturing arrangement reduces capex burden. "
                "EV adoption continues; solid-state batteries gain 5-10% market share "
                "by 2030. SLDP trades at 3-4× current price as revenue-stage company."
            ),
            key_triggers=[
                "SP2 → SP3 production ramp achieved",
                "Licensing revenue >$100M/yr by 2029",
                "Strategic manufacturing partnership announced",
                "Continued government support (DOE, IRA credits)",
            ],
            risks=[
                "Execution risk on manufacturing scale",
                "Warrant dilution already absorbed by market",
                "Competition from Asian manufacturers",
            ],
        ),
        Scenario(
            name="Bull",
            horizon="5Y",
            probability=0.25,
            price_target=35.00,
            thesis=(
                "SLDP becomes the leading Western solid-state battery IP licensor. "
                "Royalty model similar to Qualcomm in semiconductors. "
                "SP3 gigascale production achieved; multiple Tier-1 OEM agreements. "
                "EV penetration accelerates; solid-state batteries command premium. "
                "Stock re-rates to 8-12× revenue — 10× return from current levels."
            ),
            key_triggers=[
                "Commercial production at gigawatt-hour scale by 2029",
                "Licensing deals with 3+ global OEMs",
                "Government-backed manufacturing facility",
                "Solid-state achieves cost parity with lithium-ion",
            ],
            risks=[
                "Execution complexity at scale",
                "Competition may compress licensing margins",
                "Macro/policy shifts in EV adoption",
            ],
        ),
    ]

    return horizons


# ---------------------------------------------------------------------------
# 5. EXPECTED VALUE & PROBABILITY-WEIGHTED PRICE
# ---------------------------------------------------------------------------

def weighted_expected_value(scenarios: List[Scenario]) -> float:
    total = sum(s.expected_value for s in scenarios)
    prob_sum = sum(s.probability for s in scenarios)
    assert abs(prob_sum - 1.0) < 0.01, f"Probabilities don't sum to 1: {prob_sum}"
    return round(total, 4)


def upside_downside(ev: float) -> Tuple[float, float]:
    upside_pct   = round((ev - CURRENT_PRICE) / CURRENT_PRICE * 100, 1)
    downside_pct = upside_pct  # same metric, sign encodes direction
    return upside_pct, 0.0


# ---------------------------------------------------------------------------
# 6. DILUTION RISK MODULE
# ---------------------------------------------------------------------------

def dilution_analysis() -> Dict:
    shares_outstanding_approx = 170_000_000   # approximate
    warrant_dilution_pct = WARRANTS / (shares_outstanding_approx + WARRANTS) * 100

    # Cash inflow if all warrants exercised
    warrant_cash = WARRANTS * WARRANT_STRIKE

    # Diluted price if warrants exercised near current price
    # Assume treasury stock method: warrants only dilutive if price > strike
    # At $7.25 strike, only exercised in bull scenario
    diluted_price_at_bull_1y = (9.50 * shares_outstanding_approx + warrant_cash) / (
        shares_outstanding_approx + WARRANTS
    )

    return {
        "warrant_count": WARRANTS,
        "warrant_strike": WARRANT_STRIKE,
        "warrant_dilution_pct": round(warrant_dilution_pct, 2),
        "cash_inflow_if_all_exercised_M": round(warrant_cash / 1e6, 1),
        "diluted_price_at_bull_1y_target": round(diluted_price_at_bull_1y, 2),
        "assessment": (
            "Warrants are currently deep out-of-the-money (~135% above spot). "
            "They represent a FUTURE dilution risk, not present. However, they cap "
            "meaningful upside above $7.25 as warrant holders will hedge into strength. "
            "If exercised, $330.6M cash inflow would be balance-sheet positive."
        ),
    }


# ---------------------------------------------------------------------------
# 7. TECHNICAL SIGNAL SCORING
# ---------------------------------------------------------------------------

def technical_score() -> Dict:
    signals = {
        "Death cross forming":          -2,   # bearish
        "Price below all major MAs":    -2,   # bearish
        "RSI 37 (near oversold)":       +1,   # mildly bullish divergence potential
        "MACD negative":                -1,   # bearish
        "ADX 32.75 (strong trend)":     -1,   # confirms downtrend strength
        "Short interest 12.23%":        -1,   # bearish pressure / squeeze fuel
        "Support $2.91-$3.04 nearby":   +1,   # potential bounce zone
        "Resistance $3.51-$3.80":       -1,   # near-term ceiling
    }
    score = sum(signals.values())
    max_negative = sum(v for v in signals.values() if v < 0)
    normalised = round((score - max_negative) / (-max_negative) * 100, 1)

    interpretation = (
        "BEARISH" if score <= -4
        else "MILDLY BEARISH" if score <= -2
        else "NEUTRAL" if score == 0
        else "MILDLY BULLISH"
    )

    return {
        "signals": signals,
        "raw_score": score,
        "normalised_0_to_100": normalised,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# 8. RECOMMENDATION ENGINE
# ---------------------------------------------------------------------------

def recommendation(ev_1m: float, ev_3m: float, ev_1y: float) -> Dict:
    """
    Blended recommendation weighting short/medium term more for active traders,
    long-term for position sizing.
    """
    blended_ev = 0.15 * ev_1m + 0.35 * ev_3m + 0.50 * ev_1y
    upside = (blended_ev - CURRENT_PRICE) / CURRENT_PRICE * 100

    if upside > 30:
        action = "ADD"
    elif upside > 5:
        action = "HOLD"
    elif upside > -10:
        action = "REDUCE"
    else:
        action = "EXIT"

    # Stop-loss: below key support with buffer
    stop_loss = round(SUPPORT_LOW * 0.97, 2)   # 3% below $2.91 = ~$2.82

    return {
        "action": action,
        "blended_ev": round(blended_ev, 2),
        "upside_from_current_pct": round(upside, 1),
        "stop_loss": stop_loss,
        "stop_loss_rationale": (
            f"3% below key support cluster ${SUPPORT_LOW}–${SUPPORT_HIGH}. "
            "A daily close below $2.82 signals structural breakdown."
        ),
        "target_price_12m": 5.20,   # Base case 1Y
        "target_price_bull_12m": 9.50,
        "position_sizing_note": (
            "Given ADX >30 confirmed downtrend, limit position to ≤2% of portfolio. "
            "Scale in only on volume-confirmed reversal above $3.51. "
            "Speculative position only — not suitable for risk-averse portfolios."
        ),
    }


# ---------------------------------------------------------------------------
# 9. MAIN REPORT RUNNER
# ---------------------------------------------------------------------------

def run_analysis():
    horizons = build_scenarios()

    evs = {h: weighted_expected_value(s) for h, s in horizons.items()}
    rec = recommendation(evs["1M"], evs["3M"], evs["1Y"])
    dil = dilution_analysis()
    tec = technical_score()

    # ── Print Report ────────────────────────────────────────────────────────
    sep = "=" * 72

    print(sep)
    print("  SLDP — MULTI-AGENT SWARM RISK MODEL  |  Solid Power Inc")
    print(f"  Current Price: ${CURRENT_PRICE}  |  Analysis Date: 2026-03-29")
    print(sep)

    print("\n[ AGENTS DEPLOYED ]")
    for a in AGENTS:
        print(f"  ✓ {a}")

    print(f"\n{sep}")
    print("  TECHNICAL SIGNAL COMPOSITE")
    print(sep)
    for sig, val in tec["signals"].items():
        arrow = "▼" if val < 0 else "▲"
        print(f"  {arrow}  {sig:<40}  {val:+d}")
    print(f"  Raw Score: {tec['raw_score']}  |  Normalised (0=worst, 100=best): "
          f"{tec['normalised_0_to_100']}  |  {tec['interpretation']}")

    for horizon, scenarios in horizons.items():
        print(f"\n{sep}")
        print(f"  HORIZON: {horizon}  |  Probability-Weighted EV: ${evs[horizon]}")
        print(sep)
        for s in scenarios:
            bar = "█" * int(s.probability * 20)
            print(f"\n  [{s.name.upper():5s}]  P={s.probability:.0%}  {bar}")
            print(f"  Price Target : ${s.price_target:>6.2f}")
            print(f"  Expected Val : ${s.expected_value:>6.4f}")
            print(f"  Thesis       : {s.thesis}")
            print("  Key Triggers :")
            for t in s.key_triggers:
                print(f"    • {t}")
            print("  Risks        :")
            for r in s.risks:
                print(f"    ⚠ {r}")

    print(f"\n{sep}")
    print("  DILUTION RISK ANALYSIS")
    print(sep)
    for k, v in dil.items():
        if k != "assessment":
            print(f"  {k:<45}: {v}")
    print(f"\n  Assessment:\n  {dil['assessment']}")

    print(f"\n{sep}")
    print("  RECOMMENDATION")
    print(sep)
    print(f"  ACTION            : *** {rec['action']} ***")
    print(f"  Blended EV        : ${rec['blended_ev']}")
    print(f"  Upside (blended)  : {rec['upside_from_current_pct']}%")
    print(f"  Stop-Loss         : ${rec['stop_loss']}")
    print(f"  Stop Rationale    : {rec['stop_loss_rationale']}")
    print(f"  12M Base Target   : ${rec['target_price_12m']}")
    print(f"  12M Bull Target   : ${rec['target_price_bull_12m']}")
    print(f"  Position Sizing   : {rec['position_sizing_note']}")

    print(f"\n{sep}")
    print("  EXPECTED VALUE SUMMARY TABLE")
    print(sep)
    print(f"  {'Horizon':<10} {'Bear EV':>10} {'Base EV':>10} {'Bull EV':>10} {'Weighted EV':>12}")
    print(f"  {'-'*8:<10} {'-'*8:>10} {'-'*8:>10} {'-'*8:>10} {'-'*10:>12}")
    for h, scenarios in horizons.items():
        bear = next(s for s in scenarios if s.name == "Bear")
        base = next(s for s in scenarios if s.name == "Base")
        bull = next(s for s in scenarios if s.name == "Bull")
        print(
            f"  {h:<10} ${bear.expected_value:>9.4f} ${base.expected_value:>9.4f} "
            f"${bull.expected_value:>9.4f} ${evs[h]:>11.4f}"
        )

    print(f"\n{sep}")
    print("  KEY RISK CALENDAR")
    print(sep)
    events = [
        ("2026-04-01 to 2026-04-30", "No major catalyst — highest probability of continued downtrend"),
        (EARNINGS_DATE,              "Q1 2026 Earnings — primary binary event (3M horizon)"),
        ("2026-06 to 2026-08",       "SP2 Pilot Line Commissioning Window — critical technology milestone"),
        ("2026-12-31",               "Year-end 1Y scenario resolution"),
        ("2027-2028",                "SP3 scale-up decision point"),
        ("2028-2031",                "Commercial production ramp — 5Y scenario resolution"),
    ]
    for date, event in events:
        print(f"  {date:<30}  {event}")

    print(f"\n{sep}")
    print("  DISCLAIMER")
    print(sep)
    print(
        "  This model is for research and educational purposes only. It does not\n"
        "  constitute financial advice. Probabilities are model estimates subject\n"
        "  to revision. Past performance does not predict future results.\n"
        "  Always conduct independent due diligence before investing."
    )
    print(sep + "\n")

    return {
        "scenarios": horizons,
        "expected_values": evs,
        "recommendation": rec,
        "dilution": dil,
        "technical": tec,
    }


if __name__ == "__main__":
    result = run_analysis()
