#!/usr/bin/env python3
"""
Jyotish Swarm: a MiroFish-style multi-agent prediction model driven by Vedic
astrology rules instead of LLM calls (zero API cost).

MiroFish builds a world of agents (activity level, active hours, sentiment
bias, stance, influence weight), activates them stochastically each round,
lets them influence each other over a social graph (echo chamber, viral
threshold), injects "god's-eye" events, and aggregates many runs.

Here the same machinery is reused with astrological agents:

  MiroFish concept          Jyotish Swarm equivalent
  ------------------------  ------------------------------------------------
  agent                     each natal graha + transit agents (Ju/Sa/Ra/Ke)
  activity_level            dasha activation (MD 1.0, AD 0.7, PD 0.35, else 0.1)
  active_hours              the months its dasha / transit is running
  sentiment_bias            functional benefic/malefic for the Lagna + dignity
  stance (per topic)        relevance to a life area (owns/occupies/aspects/karaka)
  influence_weight          planetary strength (dignity, D9, house, digbala, SAV)
  social graph / echo       conjunctions + mutual aspects, natural friendships
  peak-hour multiplier      double transit (Jupiter + Saturn), dasha sandhi
  viral_threshold           event manifestation when area momentum crosses it
  recency_weight            exponential momentum of the area signal
  scheduled_events          ephemeris events + user-injected actions/remedies
  many simulations          Monte Carlo runs with birth-time uncertainty + noise
  ReportAgent (LLM)         Claude writes the narrative from this JSON output

This is a symbolic simulation of classical rules, not a scientifically
validated forecast. Treat the output as structured astrological reasoning.

Example
  python swarm_predict.py --date 1990-05-14 --time 10:30 --tz Asia/Kolkata \
      --lat 28.61 --lon 77.21 --months 24 --runs 1000 --time-uncertainty 10 \
      --event "2027-03:career:+0.6:launching a startup" --remedy Saturn
"""
import argparse
import json
import math
import os
import random
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vedic_chart as vc  # noqa: E402

# ----------------------------------------------------------------------------
# Life areas ("topics" the swarm debates)
# ----------------------------------------------------------------------------
AREAS = {
    "career": {"houses": {10: 1.0, 6: 0.4, 11: 0.5, 2: 0.3},
               "karakas": {"Sun": 0.6, "Saturn": 0.6, "Mercury": 0.4, "Jupiter": 0.4}, "varga": 10},
    "wealth": {"houses": {2: 1.0, 11: 1.0, 5: 0.4, 9: 0.5},
               "karakas": {"Jupiter": 0.7, "Venus": 0.5}, "varga": 2},
    "relationships": {"houses": {7: 1.0, 2: 0.4, 11: 0.3, 5: 0.5},
                      "karakas": {"Venus": 0.8, "Jupiter": 0.4}, "varga": 9},
    "health": {"houses": {1: 1.0}, "stress_houses": {6: 0.7, 8: 0.8, 12: 0.4},
               "karakas": {"Sun": 0.6, "Moon": 0.6}, "varga": 1},
    "education": {"houses": {4: 0.7, 5: 1.0, 9: 0.6, 2: 0.3},
                  "karakas": {"Mercury": 0.7, "Jupiter": 0.6}, "varga": 24},
    "property": {"houses": {4: 1.0, 11: 0.3, 2: 0.3},
                 "karakas": {"Mars": 0.6, "Venus": 0.4, "Moon": 0.3}, "varga": 4},
    "children": {"houses": {5: 1.0, 9: 0.3, 11: 0.3}, "karakas": {"Jupiter": 0.8}, "varga": 7},
    "travel_foreign": {"houses": {12: 1.0, 9: 0.6, 3: 0.4, 7: 0.3},
                       "karakas": {"Rahu": 0.7, "Moon": 0.3}, "varga": 1},
    "spirituality": {"houses": {9: 0.8, 12: 1.0, 5: 0.5, 8: 0.4},
                     "karakas": {"Ketu": 0.8, "Jupiter": 0.6, "Saturn": 0.3}, "varga": 20},
}

# Gochara: favourable houses from natal Moon
GOCHARA_GOOD = {"Jupiter": {2, 5, 7, 9, 11}, "Saturn": {3, 6, 11},
                "Rahu": {3, 6, 11}, "Ketu": {3, 6, 11}}
TRANSIT_AGENTS = ["Jupiter", "Saturn", "Rahu", "Ketu"]

NATURAL_TILT = {"Jupiter": 0.15, "Venus": 0.1, "Mercury": 0.05, "Moon": 0.05,
                "Sun": -0.05, "Mars": -0.05, "Saturn": -0.1, "Rahu": -0.1, "Ketu": -0.1}
DIGNITY_MOD = [("Exalted", 0.35), ("Moolatrikona", 0.3), ("Own sign", 0.25),
               ("Great friend", 0.15), ("Friend", 0.1), ("Neutral", 0.0),
               ("Great enemy", -0.15), ("Enemy", -0.1), ("Debilitated", -0.35)]
DIGBALA_HOUSE = {"Jupiter": 1, "Mercury": 1, "Moon": 4, "Venus": 4, "Saturn": 7, "Sun": 10, "Mars": 10}


def clamp(x, a, b):
    return max(a, min(b, x))


# ----------------------------------------------------------------------------
# Natal agent model (one per birth-time sample, cached)
# ----------------------------------------------------------------------------
def owned_houses(p, lagna_sign):
    if p in ("Rahu", "Ketu"):
        return []
    return [vc.house_from(s, lagna_sign) for s in vc.OWN[p]]


def functional_sentiment(p, lagna_sign, signs):
    """Parashari functional nature for this Lagna, in [-1, 1]."""
    if p in ("Rahu", "Ketu"):
        disp = vc.SIGN_LORD[signs[p]]
        return 0.7 * functional_sentiment(disp, lagna_sign, signs) - 0.1
    hs = owned_houses(p, lagna_sign)
    kendra = any(h in (4, 7, 10) for h in hs)
    trikona = any(h in (5, 9) for h in hs)
    if kendra and trikona:
        return 0.9  # yogakaraka
    s = 0.0
    for h in hs:
        if h in (5, 9):
            s += 0.7
        elif h == 1:
            s += 0.5
        elif h in (4, 7, 10):
            s += 0.3 if p in ("Sun", "Mars", "Saturn") else -0.1  # kendradhipati dosha
        elif h == 2:
            s += 0.0
        elif h in (3, 6):
            s += -0.35
        elif h == 11:
            s += -0.2
        elif h == 8:
            s += 0.0 if p in ("Sun", "Moon") else -0.5
        elif h == 12:
            s += -0.3
    return clamp(s, -1, 1)


def dignity_mod(text):
    for key, v in DIGNITY_MOD:
        if text.startswith(key) or key in text:
            return v
    return 0.0


def build_natal(eph, utc_dt, lat, lon):
    jd = vc.jd_from_utc(utc_dt)
    pos = eph.planets(jd)
    asc, _, _ = eph.lagna(jd, lat, lon)
    L = vc.sign_of(asc)
    signs = {p: vc.sign_of(l) for p, (l, _) in pos.items()}
    signs["Lagna"] = L
    moon_s = signs["Moon"]

    # Ashtakavarga
    bav = {}
    for p, contrib in vc.AV.items():
        pts = [0] * 12
        for c, hs in contrib.items():
            for hh in hs:
                pts[(signs[c] + hh - 1) % 12] += 1
        bav[p] = pts
    sav = [sum(bav[p][i] for p in bav) for i in range(12)]

    agents = {}
    for p in vc.PLANETS:
        l, spd = pos[p]
        s = signs[p]
        house = vc.house_from(s, L)
        dig = vc.dignity(p, l, signs)
        d9 = vc.varga_sign(l, 9)
        sent = functional_sentiment(p, L, signs) + NATURAL_TILT[p] + dignity_mod(dig)
        if d9 == s:
            sent += 0.15  # vargottama
        if p in vc.EXALT and vc.EXALT[p][0] is not None:
            if d9 == vc.EXALT[p][0] or d9 in vc.OWN.get(p, []):
                sent += 0.12
            elif d9 == (vc.EXALT[p][0] + 6) % 12:
                sent -= 0.12
        combust = False
        if p in vc.COMBUST_ORB:
            sep = abs((l - pos["Sun"][0] + 180) % 360 - 180)
            combust = sep <= vc.COMBUST_ORB[p]
            if combust:
                sent -= 0.2

        infl = 0.6 + 0.8 * dignity_mod(dig)
        infl += 0.2 if house in (1, 4, 7, 10) else 0.15 if house in (5, 9) else -0.2 if house in (6, 8, 12) else 0
        infl += 0.15 if DIGBALA_HOUSE.get(p) == house else 0
        infl += 0.1 if spd < 0 and p not in ("Rahu", "Ketu") else 0
        infl += (sav[s] - 28) / 40
        if p in bav:
            infl += (bav[p][s] - 4) / 10
        infl -= 0.15 if combust else 0
        infl = clamp(infl, 0.25, 1.6)

        # relevance to each area (the agent's "stance" on a topic)
        asp_signs = {a for a, _ in vc.aspects_from(p, s)}
        rel = {}
        for area, cfg in AREAS.items():
            r = 0.0
            for ref, w_ref in ((L, 1.0), (moon_s, 0.5)):
                own_h = [vc.house_from(x, ref) for x in vc.OWN.get(p, [])] if p not in ("Rahu", "Ketu") else []
                if p in ("Rahu", "Ketu"):
                    disp = vc.SIGN_LORD[s]
                    own_h = [vc.house_from(x, ref) for x in vc.OWN[disp]]
                    own_w = 0.5
                else:
                    own_w = 1.0
                for h, w in list(cfg["houses"].items()) + list(cfg.get("stress_houses", {}).items()):
                    hs_sign = (ref + h - 1) % 12
                    sign_factor = -1 if h in cfg.get("stress_houses", {}) else 1
                    contrib = 0
                    if h in own_h:
                        contrib += own_w
                    if s == hs_sign:
                        contrib += 0.8
                    if hs_sign in asp_signs:
                        contrib += 0.4
                    r += sign_factor * w_ref * w * contrib
            r += cfg["karakas"].get(p, 0)
            # varga confirmation
            vs = vc.varga_sign(l, cfg["varga"])
            vmod = 1.0
            if p in vc.EXALT and vc.EXALT[p][0] is not None:
                if vs == vc.EXALT[p][0] or vs in vc.OWN.get(p, []):
                    vmod = 1.15
                elif vs == (vc.EXALT[p][0] + 6) % 12:
                    vmod = 0.85
            rel[area] = clamp(r * vmod, -2.5, 2.5)
        agents[p] = {"sentiment": clamp(sent, -1, 1), "influence": round(infl, 3),
                     "relevance": rel, "sign": s, "house": house, "dignity": dig}

    # social graph: conjunction / mutual aspect, weighted by natural friendship
    links = {}
    for i, a in enumerate(vc.PLANETS):
        for b in vc.PLANETS[i + 1:]:
            rel_ab = vc.associated(a, b, signs)
            if not rel_ab or {a, b} == {"Rahu", "Ketu"}:
                continue
            fa = vc.NATURAL.get(a, {}).get(b, 0)
            fb = vc.NATURAL.get(b, {}).get(a, 0)
            links[(a, b)] = (fa + fb) / 2 if (a in vc.NATURAL and b in vc.NATURAL) else -0.3
    return {"jd": jd, "pos": pos, "signs": signs, "lagna": L, "moon_sign": moon_s,
            "agents": agents, "links": links, "bav": bav, "sav": sav,
            "dasha": vc.vimshottari(pos["Moon"][0], jd)}


# ----------------------------------------------------------------------------
# Ephemeris timeline (shared by all runs)
# ----------------------------------------------------------------------------
def month_starts(start, months):
    y, m = start.year, start.month
    out = []
    for _ in range(months):
        out.append(datetime(y, m, 15, 12, tzinfo=timezone.utc))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def build_timeline(eph, start, months):
    tl = []
    for dt in month_starts(start, months):
        jd = vc.jd_from_utc(dt)
        pos = eph.planets(jd)
        tl.append({"label": dt.strftime("%Y-%m"), "jd": jd,
                   "signs": {p: vc.sign_of(pos[p][0]) for p in TRANSIT_AGENTS},
                   "retro": {p: pos[p][1] < 0 for p in ("Jupiter", "Saturn")}})
    return tl


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
def dasha_activation(natal, jd):
    md = vc.find_period(natal["dasha"]["periods"], jd)
    ad = vc.find_period(md["antar"], jd) if md else None
    pd = vc.find_period(ad["pratyantar"], jd) if ad else None
    act = {p: 0.1 for p in vc.PLANETS}
    for per, w in ((md, 1.0), (ad, 0.7), (pd, 0.35)):
        if per:
            act[per["lord"]] = min(1.0, act[per["lord"]] + w)
    sandhi = bool(md and (md["end"] - jd < 182 or jd - md["start"] < 182))
    return act, sandhi, (md and md["lord"], ad and ad["lord"], pd and pd["lord"])


def transit_signals(natal, month, remedies):
    """Transit agents' output per area + flags (double transit, sade sati)."""
    L, M = natal["lagna"], natal["moon_sign"]
    out = {a: 0.0 for a in AREAS}
    touch = {(a, ref): set() for a in AREAS for ref in ("L", "M")}
    for p in TRANSIT_AGENTS:
        ts = month["signs"][p]
        from_moon = vc.house_from(ts, M)
        good = from_moon in GOCHARA_GOOD[p]
        base = 0.35 if good else -0.15
        if p in natal["bav"]:
            base += (natal["bav"][p][ts] - 4) / 8  # Ashtakavarga refinement
        base += (natal["sav"][ts] - 28) / 60
        if p == "Jupiter":
            base += 0.1
        if month["retro"].get(p):
            base *= 0.85
        if base < 0 and p in remedies:
            base *= 0.7
        aspected = {a for a, _ in vc.aspects_from(p, ts)} | {ts}
        for area, cfg in AREAS.items():
            hit = 0.0
            total = sum(cfg["houses"].values()) * 1.7
            for ref, w_ref, tag in ((L, 1.0, "L"), (M, 0.7, "M")):
                for h, w in cfg["houses"].items():
                    if (ref + h - 1) % 12 in aspected:
                        hit += w_ref * w
                        if w >= 1.0 and p in ("Jupiter", "Saturn"):
                            touch[(area, tag)].add(p)
            out[area] += 0.6 * base * hit / total
    double = {a: any(touch[(a, t)] >= {"Jupiter", "Saturn"} for t in ("L", "M")) for a in AREAS}
    sade = vc.house_from(month["signs"]["Saturn"], M) in (12, 1, 2)
    return out, double, sade


def area_signal(natal, row, cfg, rng, m, areas):
    """One round for one world. rng=None gives the deterministic expectation."""
    act, sandhi, lords, tsig, double, sade = row
    ag = natal["agents"]
    det = rng is None
    g = (lambda mu, sd: mu) if det else rng.gauss
    # 1) stochastic activation (MiroFish get_active_agents_for_round)
    out = {}
    for p in vc.PLANETS:
        level = clamp(act[p] + g(0, 0.05), 0, 1)
        p_active = max(level, 0.08)
        if not det and rng.random() > p_active:
            continue
        sent = ag[p]["sentiment"]
        if sent < 0 and p in cfg["remedies"]:
            sent *= 0.7
        out[p] = (sent + g(0, 0.1)) * ag[p]["influence"] * level * (p_active if det else 1)
    # 2) social influence over the aspect graph (echo chamber)
    echo = cfg["echo"]
    adj = dict(out)
    for (a, b), f in natal["links"].items():
        if a in out and b in out:
            adj[a] += echo * f * out[b] * 0.5
            adj[b] += echo * f * out[a] * 0.5
    # 3) aggregate per area
    sig, contrib = {}, {}
    for a in areas:
        s, c = 0.0, {}
        for p, v in adj.items():
            rel = ag[p]["relevance"][a]
            x = v * rel if rel >= 0 else -abs(v) * abs(rel)  # stress houses always strain
            s += x
            c[p] = x
        s = s / 3.0 + tsig[a]
        c["Transits"] = tsig[a]
        if double[a] and s > 0:
            s *= cfg["peak_multiplier"]  # double transit = MiroFish peak hour
        if sade and a in ("health", "wealth"):
            s -= 0.15 * (0.5 if ag["Saturn"]["sentiment"] > 0.3 else 1.0)
        if not det:
            for ev in cfg["events"]:
                if ev["area"] == a and m - ev["month_idx"] >= 0:
                    s += ev["delta"] * math.exp(-(m - ev["month_idx"]) / 3.0)
            s += rng.gauss(0, cfg["noise"] * (1.5 if sandhi else 1.0))
        sig[a] = s
        contrib[a] = c
    return sig, contrib


def simulate(natal_samples, timeline, cfg, rng):
    months = len(timeline)
    areas = cfg["areas"]
    R = cfg["runs"]
    sums = {a: [0.0] * months for a in areas}
    sq = {a: [0.0] * months for a in areas}
    pos_ev = {a: [0] * months for a in areas}
    neg_ev = {a: [0] * months for a in areas}
    drivers = {a: [dict() for _ in range(months)] for a in areas}
    pre = []  # per natal sample: dasha/transit per month (deterministic part)
    for natal in natal_samples:
        rows = []
        for mo in timeline:
            act, sandhi, lords = dasha_activation(natal, mo["jd"])
            tsig, double, sade = transit_signals(natal, mo, cfg["remedies"])
            rows.append((act, sandhi, lords, tsig, double, sade))
        pre.append(rows)

    # per-sample baseline ("your normal"): deterministic expectation over a 10-year window
    baselines = []
    for natal in natal_samples:
        acc = {a: [] for a in areas}
        for mo in cfg["baseline_timeline"]:
            act, sandhi, lords = dasha_activation(natal, mo["jd"])
            tsig, double, sade = transit_signals(natal, mo, cfg["remedies"])
            sig, _ = area_signal(natal, (act, sandhi, lords, tsig, double, sade), cfg, None, None, areas)
            for a in areas:
                acc[a].append(sig[a])
        baselines.append({a: (sum(v) / len(v), max(1e-6, math.sqrt(sum((x - sum(v) / len(v)) ** 2 for x in v) / len(v))))
                          for a, v in acc.items()})

    for r in range(R):
        k = rng.randrange(len(natal_samples))  # birth-time uncertainty sample
        natal, rows, base = natal_samples[k], pre[k], baselines[k]
        momentum = {a: 0.0 for a in areas}
        cooldown = {a: 0 for a in areas}
        for m in range(months):
            raw, contrib = area_signal(natal, rows[m], cfg, rng, m, areas)
            for a in areas:
                mu, sd = base[a]
                sig = (raw[a] - mu) / sd * 0.35  # relative to this person's normal
                for p, c in contrib[a].items():
                    drivers[a][m][p] = drivers[a][m].get(p, 0.0) + c
                rw = cfg["recency"]
                momentum[a] = (1 - rw) * momentum[a] + rw * sig
                sums[a][m] += sig
                sq[a][m] += sig * sig
                if cooldown[a] > 0:
                    cooldown[a] -= 1
                elif abs(momentum[a]) >= cfg["threshold"]:
                    (pos_ev if momentum[a] > 0 else neg_ev)[a][m] += 1
                    cooldown[a] = cfg["cooldown"]
                    momentum[a] *= 0.3

    res = {"months": [mo["label"] for mo in timeline], "areas": {}}
    for a in areas:
        series = []
        for m in range(months):
            mean = sums[a][m] / R
            sd = math.sqrt(max(sq[a][m] / R - mean * mean, 0))
            top = sorted(drivers[a][m].items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
            series.append({"month": timeline[m]["label"], "mean": round(mean, 3), "sd": round(sd, 3),
                           "consensus": round(abs(mean) / (sd + 1e-9), 2),
                           "p_positive_event": round(pos_ev[a][m] / R, 3),
                           "p_negative_event": round(neg_ev[a][m] / R, 3),
                           "drivers": [{"agent": p, "avg_contribution": round(v / R, 3)} for p, v in top]})
        overall = sum(s["mean"] for s in series) / months
        windows = sorted(series, key=lambda s: max(s["p_positive_event"], s["p_negative_event"]), reverse=True)[:4]
        res["areas"][a] = {"overall_mean": round(overall, 3), "series": series,
                           "top_windows": [{"month": w["month"],
                                            "polarity": "positive" if w["p_positive_event"] >= w["p_negative_event"] else "challenging",
                                            "probability": max(w["p_positive_event"], w["p_negative_event"]),
                                            "drivers": w["drivers"]} for w in windows
                                           if max(w["p_positive_event"], w["p_negative_event"]) > 0.05]}
    # dasha/transit context for the report writer (from the central sample)
    natal0 = natal_samples[len(natal_samples) // 2]
    res["context"] = [{"month": timeline[m]["label"],
                       "dasha": "/".join(x for x in pre[len(natal_samples) // 2][m][2] if x),
                       "double_transit_areas": [a for a in areas if pre[len(natal_samples) // 2][m][4][a]],
                       "sade_sati": pre[len(natal_samples) // 2][m][5],
                       "dasha_sandhi": pre[len(natal_samples) // 2][m][1]} for m in range(months)]
    res["agents"] = {p: {"house": natal0["agents"][p]["house"], "sign": vc.SIGNS[natal0["agents"][p]["sign"]],
                         "dignity": natal0["agents"][p]["dignity"],
                         "sentiment_bias": round(natal0["agents"][p]["sentiment"], 2),
                         "influence_weight": natal0["agents"][p]["influence"],
                         "stance": {a: ("supportive" if natal0["agents"][p]["relevance"][a] * natal0["agents"][p]["sentiment"] > 0.15
                                        else "opposing" if natal0["agents"][p]["relevance"][a] * natal0["agents"][p]["sentiment"] < -0.15
                                        else "observer" if abs(natal0["agents"][p]["relevance"][a]) < 0.2 else "neutral")
                                    for a in areas}}
                     for p in vc.PLANETS}
    res["natal_promise"] = {a: round(sum(natal0["agents"][p]["sentiment"] * natal0["agents"][p]["influence"]
                                         * natal0["agents"][p]["relevance"][a] for p in vc.PLANETS) / 3, 2)
                            for a in areas}
    res["social_graph"] = [{"pair": f"{a}-{b}", "friendship": f} for (a, b), f in natal0["links"].items()]
    res["lagna_samples"] = sorted({vc.SIGNS[n["lagna"]] for n in natal_samples})
    return res


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
def glyph(mean):
    return "▲▲" if mean > 0.5 else "▲ " if mean > 0.15 else "▼▼" if mean < -0.5 else "▼ " if mean < -0.15 else "· "


def render(res, cfg):
    o = ["JYOTISH SWARM SIMULATION (MiroFish-style, rule-based, zero API cost)", "=" * 78,
         f"Runs: {cfg['runs']} | Months: {len(res['months'])} | Lagna samples: {', '.join(res['lagna_samples'])}",
         f"Remedies modelled: {', '.join(cfg['remedies']) or 'none'} | Injected events: "
         + ("; ".join(f"{e['label']} {e['area']} {e['delta']:+} ({e['note']})" for e in cfg["events"]) or "none"), ""]
    o.append("AGENTS (sentiment bias / influence weight)")
    for p, d in res["agents"].items():
        o.append(f"  {p:8} H{d['house']:<2} {d['sign']:11} sent {d['sentiment_bias']:+.2f}  infl {d['influence_weight']:.2f}  {d['dignity']}")
    o.append("")
    o.append("NATAL PROMISE (what the birth chart supports overall; + favourable, - challenging)")
    o.append("  " + "  ".join(f"{a} {v:+.2f}" for a, v in res["natal_promise"].items()))
    o.append("")
    o.append("HEATMAP vs. YOUR OWN 10-YEAR NORMAL (▲▲ unusually strong, ▲ above normal, · normal, ▼ below, ▼▼ unusually low)")
    hdr = "area".ljust(15) + "".join(m[2:4] + m[5:7] + " " for m in res["months"])
    o.append(hdr + "  (YYMM)")
    for a, d in res["areas"].items():
        o.append(a.ljust(15) + "".join(glyph(s["mean"]) + "   " for s in d["series"]) + f"  avg {d['overall_mean']:+.2f}")
    o.append("")
    o.append("KEY WINDOWS (probability an event manifests across simulations)")
    for a, d in res["areas"].items():
        if not d["top_windows"]:
            continue
        ws = "; ".join(f"{w['month']} {w['polarity']} {w['probability'] * 100:.0f}% "
                       f"[{', '.join(x['agent'] for x in w['drivers'])}]" for w in d["top_windows"])
        o.append(f"  {a:15} {ws}")
    o.append("")
    o.append("TIMELINE CONTEXT")
    for c in res["context"]:
        flags = []
        if c["double_transit_areas"]:
            flags.append("double transit: " + ",".join(c["double_transit_areas"]))
        if c["sade_sati"]:
            flags.append("Sade Sati")
        if c["dasha_sandhi"]:
            flags.append("dasha sandhi")
        o.append(f"  {c['month']}  {c['dasha']:24} {' | '.join(flags)}")
    o.append("\nNote: symbolic simulation of classical Jyotish rules, not a validated scientific forecast.")
    return "\n".join(o)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="MiroFish-style swarm prediction from a Vedic chart")
    ap.add_argument("--date", required=True)
    ap.add_argument("--time", required=True)
    ap.add_argument("--tz", required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--start", help="first month YYYY-MM (default: current month)")
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--areas", default=",".join(AREAS), help="comma list of: " + ", ".join(AREAS))
    ap.add_argument("--time-uncertainty", type=float, default=0, help="± minutes of birth-time doubt")
    ap.add_argument("--event", action="append", default=[],
                    help="god's-eye injection 'YYYY-MM:area:+0.5:note' (repeatable)")
    ap.add_argument("--remedy", action="append", default=[], help="planet whose negative output is softened 30%%")
    ap.add_argument("--threshold", type=float, default=0.55, help="viral threshold for an event")
    ap.add_argument("--echo", type=float, default=0.5, help="echo-chamber strength of the aspect graph")
    ap.add_argument("--recency", type=float, default=0.4, help="momentum recency weight")
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ayanamsa", default="lahiri", choices=list(vc.AYANAMSAS))
    ap.add_argument("--node", default="mean", choices=["mean", "true"])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    eph = vc.Ephemeris(a.ayanamsa, a.node)
    tz = vc.parse_tz(a.tz)
    _, utc = vc.local_to_utc(a.date, a.time, tz)
    rng = random.Random(a.seed)

    # birth-time uncertainty: sample charts across the window (cached every 2 min)
    offsets = [0.0]
    if a.time_uncertainty > 0:
        step = 2.0
        n = int(a.time_uncertainty // step)
        offsets = [i * step for i in range(-n, n + 1)]
    from datetime import timedelta
    natal_samples = [build_natal(eph, utc + timedelta(minutes=o), a.lat, a.lon) for o in offsets]

    start = datetime.now(timezone.utc)
    if a.start:
        y, m = map(int, a.start.split("-"))
        start = datetime(y, m, 1, tzinfo=timezone.utc)
    timeline = build_timeline(eph, start, a.months)
    y0, m0 = start.year - 5, start.month
    baseline_tl = build_timeline(eph, datetime(y0, m0, 1, tzinfo=timezone.utc), 120)[::2]
    labels = [t["label"] for t in timeline]

    areas = [x.strip() for x in a.areas.split(",") if x.strip()]
    for x in areas:
        if x not in AREAS:
            raise SystemExit(f"unknown area {x}; choose from {list(AREAS)}")
    events = []
    for e in a.event:
        parts = e.split(":", 3)
        lab, area, delta = parts[0], parts[1], float(parts[2])
        if lab not in labels:
            raise SystemExit(f"event month {lab} outside simulated range {labels[0]}..{labels[-1]}")
        events.append({"label": lab, "month_idx": labels.index(lab), "area": area,
                       "delta": delta, "note": parts[3] if len(parts) > 3 else ""})
    remedies = [r.capitalize() for r in a.remedy]

    cfg = {"runs": a.runs, "areas": areas, "events": events, "remedies": remedies,
           "threshold": a.threshold, "echo": a.echo, "recency": a.recency, "noise": a.noise,
           "peak_multiplier": 1.3, "cooldown": 6, "baseline_timeline": baseline_tl}
    res = simulate(natal_samples, timeline, cfg, rng)
    res["config"] = {k: v for k, v in cfg.items() if k != "baseline_timeline"}
    res["disclaimer"] = "Symbolic simulation of classical Jyotish rules; not a scientifically validated forecast."
    print(json.dumps(res, indent=1) if a.json else render(res, cfg))


if __name__ == "__main__":
    main()
