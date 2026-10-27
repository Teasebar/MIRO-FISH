#!/usr/bin/env python3
"""
Vedic (Jyotish) calculation engine for the vedic-astrology skill.

Uses the Swiss Ephemeris (pyswisseph), the same engine behind Jagannatha Hora,
Kala and most professional Jyotish software. Sidereal zodiac, Lahiri
(Chitrapaksha) ayanamsa by default, whole-sign houses (Parashari standard).

Subcommands
  chart     Full natal chart: D1 + 15 vargas, nakshatras, dignities, aspects,
            Vimshottari dasha, panchang at birth, Jaimini karakas & arudhas,
            Ashtakavarga, yogas/doshas, current transits & Sade Sati.
  match     Ashtakoota (Guna Milan, 36 points) compatibility + Mangal dosha.
  panchang  Daily panchang for a date/place: sunrise/sunset, tithi, nakshatra,
            yoga, karana, Rahu Kaal, Yamaganda, Gulika, Abhijit muhurta.

Examples
  python vedic_chart.py chart --date 1990-05-14 --time 10:30 --tz Asia/Kolkata \
      --lat 28.6139 --lon 77.2090 --name "Asha"
  python vedic_chart.py match --boy "1990-05-14,10:30,Asia/Kolkata,28.61,77.21" \
      --girl "1992-08-02,06:15,Asia/Kolkata,19.07,72.88"
  python vedic_chart.py panchang --date 2026-10-27 --tz Asia/Kolkata --lat 28.61 --lon 77.21

Add --json to any subcommand for machine-readable output.
"""
import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None

try:
    import swisseph as swe
except ImportError:
    sys.stderr.write(
        "pyswisseph is not installed. Run: pip install pyswisseph\n"
        "If the build fails with 'install_layout', use a virtualenv:\n"
        "  python3 -m venv /tmp/jyotish && /tmp/jyotish/bin/pip install pyswisseph\n"
        "  /tmp/jyotish/bin/python vedic_chart.py ...\n")
    sys.exit(2)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra",
         "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
SIGNS_SKT = ["Mesha", "Vrishabha", "Mithuna", "Karka", "Simha", "Kanya", "Tula",
             "Vrischika", "Dhanu", "Makara", "Kumbha", "Meena"]
SIGN_ABBR = ["Ar", "Ta", "Ge", "Cn", "Le", "Vi", "Li", "Sc", "Sg", "Cp", "Aq", "Pi"]
SIGN_LORD = ["Mars", "Venus", "Mercury", "Moon", "Sun", "Mercury", "Venus",
             "Mars", "Jupiter", "Saturn", "Saturn", "Jupiter"]

PLANETS = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]
SEVEN = PLANETS[:7]
ABBR = {"Sun": "Su", "Moon": "Mo", "Mars": "Ma", "Mercury": "Me", "Jupiter": "Ju",
        "Venus": "Ve", "Saturn": "Sa", "Rahu": "Ra", "Ketu": "Ke", "Lagna": "As"}
SWE_ID = {"Sun": swe.SUN, "Moon": swe.MOON, "Mars": swe.MARS, "Mercury": swe.MERCURY,
          "Jupiter": swe.JUPITER, "Venus": swe.VENUS, "Saturn": swe.SATURN}

NAKSHATRAS = ["Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
              "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni",
              "Uttara Phalguni", "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha",
              "Jyeshtha", "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana",
              "Dhanishta", "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"]
NAK_SPAN = 360.0 / 27

DASHA_ORDER = ["Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury"]
DASHA_YEARS = {"Ketu": 7, "Venus": 20, "Sun": 6, "Moon": 10, "Mars": 7, "Rahu": 18,
               "Jupiter": 16, "Saturn": 19, "Mercury": 17}

EXALT = {"Sun": (0, 10), "Moon": (1, 3), "Mars": (9, 28), "Mercury": (5, 15),
         "Jupiter": (3, 5), "Venus": (11, 27), "Saturn": (6, 20),
         "Rahu": (1, None), "Ketu": (7, None)}  # nodes: common (not universal) view
OWN = {"Sun": [4], "Moon": [3], "Mars": [0, 7], "Mercury": [2, 5], "Jupiter": [8, 11],
       "Venus": [1, 6], "Saturn": [9, 10], "Rahu": [10], "Ketu": [7]}
MOOLATRIKONA = {"Sun": (4, 0, 20), "Moon": (1, 4, 30), "Mars": (0, 0, 12),
                "Mercury": (5, 16, 20), "Jupiter": (8, 0, 10), "Venus": (6, 0, 15),
                "Saturn": (10, 0, 20)}

# Naisargika (natural) relationships, BPHS: 1 friend, 0 neutral, -1 enemy
NATURAL = {
    "Sun": {"Moon": 1, "Mars": 1, "Jupiter": 1, "Mercury": 0, "Venus": -1, "Saturn": -1},
    "Moon": {"Sun": 1, "Mercury": 1, "Mars": 0, "Jupiter": 0, "Venus": 0, "Saturn": 0},
    "Mars": {"Sun": 1, "Moon": 1, "Jupiter": 1, "Venus": 0, "Saturn": 0, "Mercury": -1},
    "Mercury": {"Sun": 1, "Venus": 1, "Mars": 0, "Jupiter": 0, "Saturn": 0, "Moon": -1},
    "Jupiter": {"Sun": 1, "Moon": 1, "Mars": 1, "Saturn": 0, "Mercury": -1, "Venus": -1},
    "Venus": {"Mercury": 1, "Saturn": 1, "Mars": 0, "Jupiter": 0, "Sun": -1, "Moon": -1},
    "Saturn": {"Mercury": 1, "Venus": 1, "Jupiter": 0, "Sun": -1, "Moon": -1, "Mars": -1},
}
COMPOUND_NAME = {2: "Great friend (Adhimitra)", 1: "Friend (Mitra)", 0: "Neutral (Sama)",
                 -1: "Enemy (Shatru)", -2: "Great enemy (Adhishatru)"}

COMBUST_ORB = {"Moon": 12, "Mars": 17, "Mercury": 14, "Jupiter": 11, "Venus": 10, "Saturn": 15}
COMBUST_ORB_RETRO = {"Mercury": 12, "Venus": 8}

NATURAL_BENEFICS = {"Jupiter", "Venus", "Mercury", "Moon"}  # Moon/Mercury conditional

TITHIS = ["Pratipada", "Dwitiya", "Tritiya", "Chaturthi", "Panchami", "Shashthi",
          "Saptami", "Ashtami", "Navami", "Dashami", "Ekadashi", "Dwadashi",
          "Trayodashi", "Chaturdashi"]
YOGAS_27 = ["Vishkambha", "Priti", "Ayushman", "Saubhagya", "Shobhana", "Atiganda",
            "Sukarma", "Dhriti", "Shula", "Ganda", "Vriddhi", "Dhruva", "Vyaghata",
            "Harshana", "Vajra", "Siddhi", "Vyatipata", "Variyan", "Parigha", "Shiva",
            "Siddha", "Sadhya", "Shubha", "Shukla", "Brahma", "Indra", "Vaidhriti"]
KARANAS_MOVABLE = ["Bava", "Balava", "Kaulava", "Taitila", "Gara", "Vanija", "Vishti (Bhadra)"]
VARA = ["Somavara (Monday)", "Mangalavara (Tuesday)", "Budhavara (Wednesday)",
        "Guruvara (Thursday)", "Shukravara (Friday)", "Shanivara (Saturday)",
        "Ravivara (Sunday)"]  # indexed by Python weekday()
VARA_LORD = ["Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Sun"]

AYANAMSAS = {"lahiri": swe.SIDM_LAHIRI, "raman": swe.SIDM_RAMAN,
             "kp": swe.SIDM_KRISHNAMURTI, "true-chitra": swe.SIDM_TRUE_CITRA,
             "yukteshwar": swe.SIDM_YUKTESHWAR, "fagan": swe.SIDM_FAGAN_BRADLEY}

# Ashtakavarga benefic houses (BPHS). AV[planet][contributor] = houses counted
# from the contributor's sign.
AV = {
    "Sun": {"Sun": [1, 2, 4, 7, 8, 9, 10, 11], "Moon": [3, 6, 10, 11],
            "Mars": [1, 2, 4, 7, 8, 9, 10, 11], "Mercury": [3, 5, 6, 9, 10, 11, 12],
            "Jupiter": [5, 6, 9, 11], "Venus": [6, 7, 12],
            "Saturn": [1, 2, 4, 7, 8, 9, 10, 11], "Lagna": [3, 4, 6, 10, 11, 12]},
    "Moon": {"Sun": [3, 6, 7, 8, 10, 11], "Moon": [1, 3, 6, 7, 10, 11],
             "Mars": [2, 3, 5, 6, 9, 10, 11], "Mercury": [1, 3, 4, 5, 7, 8, 10, 11],
             "Jupiter": [1, 4, 7, 8, 10, 11, 12], "Venus": [3, 4, 5, 7, 9, 10, 11],
             "Saturn": [3, 5, 6, 11], "Lagna": [3, 6, 10, 11]},
    "Mars": {"Sun": [3, 5, 6, 10, 11], "Moon": [3, 6, 11], "Mars": [1, 2, 4, 7, 8, 10, 11],
             "Mercury": [3, 5, 6, 11], "Jupiter": [6, 10, 11, 12], "Venus": [6, 8, 11, 12],
             "Saturn": [1, 4, 7, 8, 9, 10, 11], "Lagna": [1, 3, 6, 10, 11]},
    "Mercury": {"Sun": [5, 6, 9, 11, 12], "Moon": [2, 4, 6, 8, 10, 11],
                "Mars": [1, 2, 4, 7, 8, 9, 10, 11], "Mercury": [1, 3, 5, 6, 9, 10, 11, 12],
                "Jupiter": [6, 8, 11, 12], "Venus": [1, 2, 3, 4, 5, 8, 9, 11],
                "Saturn": [1, 2, 4, 7, 8, 9, 10, 11], "Lagna": [1, 2, 4, 6, 8, 10, 11]},
    "Jupiter": {"Sun": [1, 2, 3, 4, 7, 8, 9, 10, 11], "Moon": [2, 5, 7, 9, 11],
                "Mars": [1, 2, 4, 7, 8, 10, 11], "Mercury": [1, 2, 4, 5, 6, 9, 10, 11],
                "Jupiter": [1, 2, 3, 4, 7, 8, 10, 11], "Venus": [2, 5, 6, 9, 10, 11],
                "Saturn": [3, 5, 6, 12], "Lagna": [1, 2, 4, 5, 6, 7, 9, 10, 11]},
    "Venus": {"Sun": [8, 11, 12], "Moon": [1, 2, 3, 4, 5, 8, 9, 11, 12],
              "Mars": [3, 5, 6, 9, 11, 12], "Mercury": [3, 5, 6, 9, 11],
              "Jupiter": [5, 8, 9, 10, 11], "Venus": [1, 2, 3, 4, 5, 8, 9, 10, 11],
              "Saturn": [3, 4, 5, 8, 9, 10, 11], "Lagna": [1, 2, 3, 4, 5, 8, 9, 11]},
    "Saturn": {"Sun": [1, 2, 4, 7, 8, 10, 11], "Moon": [3, 6, 11],
               "Mars": [3, 5, 6, 10, 11, 12], "Mercury": [6, 8, 9, 10, 11, 12],
               "Jupiter": [5, 6, 11, 12], "Venus": [6, 11, 12], "Saturn": [3, 5, 6, 11],
               "Lagna": [1, 3, 4, 6, 10, 11]},
}

# Ashtakoota tables
VARNA_RANK = {3: 4, 7: 4, 11: 4, 0: 3, 4: 3, 8: 3, 1: 2, 5: 2, 9: 2, 2: 1, 6: 1, 10: 1}
VARNA_NAME = {4: "Brahmin", 3: "Kshatriya", 2: "Vaishya", 1: "Shudra"}
VASHYA_NAMES = ["Chatushpada", "Manava", "Jalachara", "Vanachara", "Keeta"]
VASHYA_TABLE = [  # rows boy, cols girl
    [2, 1, 1, 0, 1],
    [1, 2, 0.5, 0, 1],
    [1, 0.5, 2, 1, 1],
    [0.5, 0, 1, 2, 0],
    [1, 1, 1, 0, 2],
]
YONI_ANIMALS = ["Horse", "Elephant", "Sheep", "Serpent", "Dog", "Cat", "Rat", "Cow",
                "Buffalo", "Tiger", "Deer", "Monkey", "Mongoose", "Lion"]
NAK_YONI = [0, 1, 2, 3, 3, 4, 5, 2, 5, 6, 6, 7, 8, 9, 8, 9, 10, 10, 4, 11, 12, 11,
            13, 0, 13, 7, 1]
YONI_TABLE = [
    [4, 2, 2, 3, 2, 2, 2, 1, 0, 1, 3, 3, 2, 1],
    [2, 4, 3, 3, 2, 2, 2, 2, 3, 1, 2, 3, 2, 0],
    [2, 3, 4, 2, 1, 2, 1, 3, 3, 1, 2, 0, 3, 1],
    [3, 3, 2, 4, 2, 1, 1, 1, 1, 2, 2, 2, 0, 2],
    [2, 2, 1, 2, 4, 2, 1, 2, 2, 1, 0, 2, 1, 1],
    [2, 2, 2, 1, 2, 4, 0, 2, 2, 1, 3, 3, 2, 1],
    [2, 2, 1, 1, 1, 0, 4, 2, 2, 2, 2, 2, 1, 2],
    [1, 2, 3, 1, 2, 2, 2, 4, 3, 0, 3, 2, 2, 1],
    [0, 3, 3, 1, 2, 2, 2, 3, 4, 1, 2, 2, 2, 1],
    [1, 1, 1, 2, 1, 1, 2, 0, 1, 4, 1, 1, 2, 1],
    [3, 2, 2, 2, 0, 3, 2, 3, 2, 1, 4, 2, 2, 1],
    [3, 3, 0, 2, 2, 3, 2, 2, 2, 1, 2, 4, 3, 2],
    [2, 2, 3, 0, 1, 2, 1, 2, 2, 2, 2, 3, 4, 2],
    [1, 0, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 2, 4],
]
GANA_NAMES = ["Deva", "Manushya", "Rakshasa"]
NAK_GANA = [0, 1, 2, 1, 0, 1, 0, 0, 2, 2, 1, 1, 0, 2, 0, 2, 0, 2, 2, 1, 1, 0, 2, 2, 1, 1, 0]
GANA_TABLE = [[6, 6, 1], [5, 6, 0], [1, 0, 6]]  # rows boy, cols girl
NADI_NAMES = ["Adi (Vata)", "Madhya (Pitta)", "Antya (Kapha)"]
NADI_CYCLE = [0, 1, 2, 2, 1, 0]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def norm(x):
    return x % 360.0


def dms(deg):
    deg = deg % 360.0
    d = int(deg)
    m_f = (deg - d) * 60
    m = int(m_f)
    s = int(round((m_f - m) * 60))
    if s == 60:
        s, m = 0, m + 1
    if m == 60:
        m, d = 0, d + 1
    return f"{d:02d}°{m:02d}'{s:02d}\""


def sign_of(lon):
    return int(norm(lon) // 30)


def house_from(sign, ref_sign):
    return (sign - ref_sign) % 12 + 1


def parse_tz(tz):
    """Accept an IANA name ('Asia/Kolkata') or a fixed offset ('+05:30', '-4')."""
    m = re.fullmatch(r"(?:UTC|GMT)?\s*([+-])(\d{1,2})(?::?(\d{2}))?", tz.strip())
    if m:
        sign = 1 if m.group(1) == "+" else -1
        off = timedelta(hours=int(m.group(2)), minutes=int(m.group(3) or 0))
        return timezone(sign * off)
    if ZoneInfo is None:
        raise SystemExit("zoneinfo unavailable; pass a fixed offset like +05:30")
    return ZoneInfo(tz)


def local_to_utc(date_s, time_s, tz):
    t = time_s if time_s.count(":") == 2 else time_s + ":00"
    naive = datetime.strptime(f"{date_s} {t}", "%Y-%m-%d %H:%M:%S")
    local = naive.replace(tzinfo=tz)
    return local, local.astimezone(timezone.utc)


def jd_from_utc(dt):
    h = dt.hour + dt.minute / 60 + dt.second / 3600 + dt.microsecond / 3.6e9
    return swe.julday(dt.year, dt.month, dt.day, h)


def utc_from_jd(jd):
    y, m, d, h = swe.revjul(jd)
    return datetime(y, m, d, tzinfo=timezone.utc) + timedelta(hours=h)


def fmt_local(jd, tz, with_time=False):
    dt = utc_from_jd(jd).astimezone(tz)
    return dt.strftime("%Y-%m-%d %H:%M" if with_time else "%Y-%m-%d")


def nakshatra_info(lon):
    lon = norm(lon)
    idx = int(lon // NAK_SPAN)
    pada = int((lon % NAK_SPAN) // (NAK_SPAN / 4)) + 1
    return {"index": idx, "name": NAKSHATRAS[idx], "pada": pada,
            "lord": DASHA_ORDER[idx % 9],
            "fraction_elapsed": (lon % NAK_SPAN) / NAK_SPAN}


# ----------------------------------------------------------------------------
# Ephemeris
# ----------------------------------------------------------------------------
class Ephemeris:
    def __init__(self, ayanamsa="lahiri", node="mean"):
        swe.set_sid_mode(AYANAMSAS[ayanamsa])
        ephe = os.environ.get("SE_EPHE_PATH")
        if ephe:
            swe.set_ephe_path(ephe)
            self.base = swe.FLG_SWIEPH
        else:
            self.base = swe.FLG_MOSEPH  # built-in, no data files; sub-arcsecond for planets
        self.flags = self.base | swe.FLG_SIDEREAL | swe.FLG_SPEED
        self.node_id = swe.MEAN_NODE if node == "mean" else swe.TRUE_NODE
        self.ayanamsa_name = ayanamsa
        self.node = node

    def planets(self, jd):
        out = {}
        for p, pid in SWE_ID.items():
            xx, _ = swe.calc_ut(jd, pid, self.flags)
            out[p] = (norm(xx[0]), xx[3])
        xx, _ = swe.calc_ut(jd, self.node_id, self.flags)
        out["Rahu"] = (norm(xx[0]), xx[3])
        out["Ketu"] = (norm(xx[0] + 180), xx[3])
        return out

    def lagna(self, jd, lat, lon):
        cusps, ascmc = swe.houses_ex(jd, lat, lon, b"O", swe.FLG_SIDEREAL | self.base)
        return norm(ascmc[0]), norm(ascmc[1]), [norm(c) for c in cusps[:12]]

    def ayanamsa(self, jd):
        return swe.get_ayanamsa_ut(jd)

    def sun_event(self, jd_start, lat, lon, rise=True):
        flag = swe.CALC_RISE if rise else swe.CALC_SET
        res, tret = swe.rise_trans(jd_start, swe.SUN, flag, (lon, lat, 0), 0, 0, self.base)
        return tret[0] if res == 0 else None


# ----------------------------------------------------------------------------
# Divisional charts (Parashari)
# ----------------------------------------------------------------------------
def varga_sign(lon, n):
    lon = norm(lon)
    s = int(lon // 30)
    d = lon % 30
    odd = s % 2 == 0  # Aries (index 0) is an odd sign
    mov, fix = s % 3 == 0, s % 3 == 1
    if n == 1:
        return s
    if n == 2:  # Hora: odd 0-15 Leo, 15-30 Cancer; even reversed
        first = d < 15
        return 4 if first == odd else 3
    if n == 3:
        return (s + 4 * int(d // 10)) % 12
    if n == 4:
        return (s + 3 * int(d // 7.5)) % 12
    if n == 7:
        part = int(d // (30 / 7))
        return (s + part) % 12 if odd else (s + 6 + part) % 12
    if n == 9:
        return int(lon // (30 / 9)) % 12
    if n == 10:
        part = int(d // 3)
        return (s + part) % 12 if odd else (s + 8 + part) % 12
    if n == 12:
        return (s + int(d // 2.5)) % 12
    if n == 16:
        start = 0 if mov else 4 if fix else 8
        return (start + int(d // (30 / 16))) % 12
    if n == 20:
        start = 0 if mov else 8 if fix else 4
        return (start + int(d // 1.5)) % 12
    if n == 24:
        start = 4 if odd else 3
        return (start + int(d // 1.25)) % 12
    if n == 27:
        return int(lon // (30 / 27)) % 12
    if n == 30:
        if odd:
            for lim, sg in ((5, 0), (10, 10), (18, 8), (25, 2), (30, 6)):
                if d < lim:
                    return sg
        for lim, sg in ((5, 1), (12, 5), (20, 11), (25, 9), (30, 7)):
            if d < lim:
                return sg
    if n == 40:
        start = 0 if odd else 6
        return (start + int(d // 0.75)) % 12
    if n == 45:
        start = 0 if mov else 4 if fix else 8
        return (start + int(d // (30 / 45))) % 12
    if n == 60:
        return (s + int(d // 0.5)) % 12
    raise ValueError(n)


VARGAS = {1: "Rashi (body, overall life)", 2: "Hora (wealth)", 3: "Drekkana (siblings, courage)",
          4: "Chaturthamsa (property, fortune)", 7: "Saptamsa (children)",
          9: "Navamsa (spouse, dharma, inner strength)", 10: "Dasamsa (career)",
          12: "Dwadasamsa (parents)", 16: "Shodasamsa (vehicles, comforts)",
          20: "Vimsamsa (spiritual practice)", 24: "Chaturvimsamsa (education)",
          27: "Saptavimsamsa (strength, stamina)", 30: "Trimsamsa (misfortune, character)",
          40: "Khavedamsa (auspicious/inauspicious effects)",
          45: "Akshavedamsa (general well-being)", 60: "Shashtiamsa (past karma; needs exact time)"}


# ----------------------------------------------------------------------------
# Chart building
# ----------------------------------------------------------------------------
def compound_relation(p, other, signs):
    """Panchadha maitri of planet p towards `other` (both among the 7 grahas)."""
    if p == other:
        return 2
    nat = NATURAL[p].get(other, 0)
    h = house_from(signs[other], signs[p])
    temp = 1 if h in (2, 3, 4, 10, 11, 12) else -1
    return nat + temp


def dignity(p, lon, signs):
    s, d = sign_of(lon), lon % 30
    if p in EXALT:
        es, deg = EXALT[p]
        if s == es:
            return "Exalted" + (f" (deep exaltation {deg}°)" if deg is not None and abs(d - deg) < 1 else "")
        if s == (es + 6) % 12:
            return "Debilitated" + (f" (deepest {deg}°)" if deg is not None and abs(d - deg) < 1 else "")
    if p in MOOLATRIKONA:
        ms, a, b = MOOLATRIKONA[p]
        if s == ms and a <= d < b:
            return "Moolatrikona"
    if s in OWN.get(p, []):
        return "Own sign"
    if p in SEVEN:
        lord = SIGN_LORD[s]
        return f"{COMPOUND_NAME[compound_relation(p, lord, signs)]}'s sign ({lord})"
    return f"In {SIGN_LORD[s]}'s sign"


def aspects_from(p, sign):
    """Whole-sign graha drishti: returns list of (aspected_sign, aspect_house_number)."""
    offs = [7]
    if p == "Mars":
        offs = [4, 7, 8]
    elif p == "Jupiter":
        offs = [5, 7, 9]
    elif p == "Saturn":
        offs = [3, 7, 10]
    elif p in ("Rahu", "Ketu"):
        offs = [5, 7, 9]  # per some authorities
    return [((sign + o - 1) % 12, o) for o in offs]


def vimshottari(moon_lon, birth_jd, year_days=365.25):
    nk = nakshatra_info(moon_lon)
    lord = nk["lord"]
    start_idx = DASHA_ORDER.index(lord)
    balance_years = DASHA_YEARS[lord] * (1 - nk["fraction_elapsed"])
    # Start of the birth MD (in the past) so sub-periods line up
    md_start = birth_jd - (DASHA_YEARS[lord] - balance_years) * year_days
    periods = []
    t = md_start
    for i in range(10):  # covers > 120 years from birth
        md = DASHA_ORDER[(start_idx + i) % 9]
        dur = DASHA_YEARS[md] * year_days
        ads = []
        at = t
        for j in range(9):
            ad = DASHA_ORDER[(DASHA_ORDER.index(md) + j) % 9]
            adur = dur * DASHA_YEARS[ad] / 120
            pds = []
            pt = at
            for k in range(9):
                pd = DASHA_ORDER[(DASHA_ORDER.index(ad) + k) % 9]
                pdur = adur * DASHA_YEARS[pd] / 120
                pds.append({"lord": pd, "start": pt, "end": pt + pdur})
                pt += pdur
            ads.append({"lord": ad, "start": at, "end": at + adur, "pratyantar": pds})
            at += adur
        periods.append({"lord": md, "start": t, "end": t + dur, "antar": ads})
        t += dur
    return {"balance_years": balance_years, "birth_lord": lord, "periods": periods}


def find_period(periods, jd):
    for p in periods:
        if p["start"] <= jd < p["end"]:
            return p
    return None


def build_chart(eph, local_dt, utc_dt, lat, lon, transit_utc=None, year_days=365.25):
    jd = jd_from_utc(utc_dt)
    pos = eph.planets(jd)
    asc, mc, cusps = eph.lagna(jd, lat, lon)
    lagna_sign = sign_of(asc)
    signs = {p: sign_of(l) for p, (l, _) in pos.items()}
    signs["Lagna"] = lagna_sign

    sun_lon = pos["Sun"][0]
    planets = {}
    for p in PLANETS:
        l, spd = pos[p]
        s = sign_of(l)
        retro = spd < 0 if p not in ("Rahu", "Ketu") else True
        combust = False
        if p in COMBUST_ORB:
            orb = COMBUST_ORB_RETRO.get(p, COMBUST_ORB[p]) if retro else COMBUST_ORB[p]
            sep = abs((l - sun_lon + 180) % 360 - 180)
            combust = sep <= orb
        # Bhava chalit (Sripati/Porphyry cusps as bhava madhya boundaries)
        chalit = None
        for i in range(12):
            a, b = cusps[i], cusps[(i + 1) % 12]
            if (l - a) % 360 < (b - a) % 360:
                chalit = i + 1
                break
        planets[p] = {
            "longitude": round(l, 6), "sign": SIGNS[s], "sign_index": s,
            "degree_in_sign": dms(l % 30), "house": house_from(s, lagna_sign),
            "bhava_chalit_house": chalit, "retrograde": retro, "combust": combust,
            "speed": round(spd, 4), "nakshatra": nakshatra_info(l),
            "dignity": dignity(p, l, signs), "sign_lord": SIGN_LORD[s],
            "vargas": {f"D{n}": SIGNS[varga_sign(l, n)] for n in VARGAS},
        }
    lagna = {"longitude": round(asc, 6), "sign": SIGNS[lagna_sign], "sign_index": lagna_sign,
             "degree_in_sign": dms(asc % 30), "nakshatra": nakshatra_info(asc),
             "lord": SIGN_LORD[lagna_sign],
             "vargas": {f"D{n}": SIGNS[varga_sign(asc, n)] for n in VARGAS},
             "midheaven_sidereal": dms(mc) + f" {SIGNS[sign_of(mc)]}"}

    # House lords and occupants
    houses = []
    for h in range(1, 13):
        s = (lagna_sign + h - 1) % 12
        lord = SIGN_LORD[s]
        aspected_by = [p for p in PLANETS for (asg, _) in aspects_from(p, signs[p]) if asg == s]
        houses.append({"house": h, "sign": SIGNS[s], "lord": lord,
                       "lord_placed_in_house": planets[lord]["house"],
                       "occupants": [p for p in PLANETS if signs[p] == s],
                       "aspected_by": aspected_by})

    # Jaimini chara karakas (7-karaka scheme)
    kar_names = ["Atmakaraka", "Amatyakaraka", "Bhratrikaraka", "Matrikaraka",
                 "Putrakaraka", "Gnatikaraka", "Darakaraka"]
    order = sorted(SEVEN, key=lambda p: pos[p][0] % 30, reverse=True)
    karakas = {kar_names[i]: order[i] for i in range(7)}

    # Arudha padas (A1..A12); exception: if pada falls in 1st/7th from the house, take 10th from it
    arudhas = {}
    for h in range(1, 13):
        hs = (lagna_sign + h - 1) % 12
        lord = SIGN_LORD[hs]
        cnt = (signs[lord] - hs) % 12
        pada = (signs[lord] + cnt) % 12
        if (pada - hs) % 12 in (0, 6):
            pada = (pada + 9) % 12
        label = "AL (Arudha Lagna)" if h == 1 else "UL (Upapada)" if h == 12 else f"A{h}"
        arudhas[label] = {"sign": SIGNS[pada], "house": house_from(pada, lagna_sign)}

    # Ashtakavarga
    bav = {}
    for p, contrib in AV.items():
        pts = [0] * 12
        for c, hs in contrib.items():
            for hh in hs:
                pts[(signs[c] + hh - 1) % 12] += 1
        bav[p] = pts
    sav = [sum(bav[p][i] for p in bav) for i in range(12)]
    assert sum(sav) == 337, "Ashtakavarga integrity check failed"

    # Panchang at birth
    moon_lon = pos["Moon"][0]
    diff = norm(moon_lon - sun_lon)
    panchang = panchang_elements(sun_lon, moon_lon)
    sunrise = eph.sun_event(jd - 1, lat, lon, True)
    # find the sunrise on/before birth
    sr_before = None
    t0 = jd - 1.2
    while True:
        sr = eph.sun_event(t0, lat, lon, True)
        if sr is None or sr > jd:
            break
        sr_before = sr
        t0 = sr + 0.01
    vara_date = local_dt.date()
    if sr_before is not None:
        vara_date = utc_from_jd(sr_before).astimezone(local_dt.tzinfo).date()
    panchang["vara"] = VARA[vara_date.weekday()]
    panchang["vara_lord"] = VARA_LORD[vara_date.weekday()]
    panchang["note"] = "Vedic day runs sunrise to sunrise; vara reflects that."
    if sr_before is not None:
        panchang["sunrise_before_birth"] = fmt_local(sr_before, local_dt.tzinfo, True)

    # Dasha
    vd = vimshottari(moon_lon, jd, year_days)

    # Birth-time sensitivity
    def lagna_sign_at(minutes, n=1):
        a, _, _ = eph.lagna(jd + minutes / 1440, lat, lon)
        return varga_sign(a, n)

    def minutes_to_change(n, direction):
        base = lagna_sign_at(0, n)
        for m in range(1, 241):
            if lagna_sign_at(direction * m, n) != base:
                return m
        return None

    sensitivity = {
        "lagna_changes_in_minutes": {"earlier": minutes_to_change(1, -1), "later": minutes_to_change(1, 1)},
        "navamsa_lagna_changes_in_minutes": {"earlier": minutes_to_change(9, -1), "later": minutes_to_change(9, 1)},
        "moon_nakshatra_boundary_degrees": round(min(moon_lon % NAK_SPAN, NAK_SPAN - moon_lon % NAK_SPAN), 3),
    }

    chart = {
        "input": {"local_time": local_dt.isoformat(), "utc": utc_dt.isoformat(),
                  "latitude": lat, "longitude": lon},
        "settings": {"ayanamsa": eph.ayanamsa_name, "ayanamsa_value": dms(eph.ayanamsa(jd)),
                     "node": eph.node, "houses": "whole sign (bhava chalit uses Sripati cusps)",
                     "dasha_year_days": year_days,
                     "ephemeris": "Swiss Ephemeris files" if eph.base == swe.FLG_SWIEPH else "Swiss Ephemeris (Moshier built-in)"},
        "julian_day_ut": jd,
        "lagna": lagna, "planets": planets, "houses": houses,
        "karakas": karakas, "arudhas": arudhas,
        "ashtakavarga": {"bav": {p: dict(zip(SIGNS, v)) for p, v in bav.items()},
                         "sav": dict(zip(SIGNS, sav)),
                         "sav_by_house": {h: sav[(lagna_sign + h - 1) % 12] for h in range(1, 13)}},
        "panchang": panchang,
        "time_sensitivity": sensitivity,
    }
    chart["yogas"] = detect_yogas(chart, pos, signs, lagna_sign)
    chart["doshas"] = detect_doshas(pos, signs, lagna_sign)

    # Dasha summary + transit
    tr_utc = transit_utc or datetime.now(timezone.utc)
    tr_jd = jd_from_utc(tr_utc)
    tz = local_dt.tzinfo
    md = find_period(vd["periods"], tr_jd)
    ad = find_period(md["antar"], tr_jd) if md else None
    pd = find_period(ad["pratyantar"], tr_jd) if ad else None
    chart["vimshottari"] = {
        "birth_nakshatra_lord": vd["birth_lord"],
        "balance_at_birth": f"{vd['birth_lord']} {years_str(vd['balance_years'])}",
        "mahadashas": [{"lord": p["lord"], "start": fmt_local(max(p["start"], jd), tz),
                        "end": fmt_local(p["end"], tz)} for p in vd["periods"]
                       if p["end"] > jd and p["start"] < jd + 120 * year_days],
        "as_of": tr_utc.astimezone(tz).strftime("%Y-%m-%d"),
        "current": {
            "mahadasha": md and {"lord": md["lord"], "start": fmt_local(md["start"], tz), "end": fmt_local(md["end"], tz)},
            "antardasha": ad and {"lord": ad["lord"], "start": fmt_local(ad["start"], tz), "end": fmt_local(ad["end"], tz)},
            "pratyantar": pd and {"lord": pd["lord"], "start": fmt_local(pd["start"], tz), "end": fmt_local(pd["end"], tz)},
        },
        "antardashas_current_md": md and [{"lord": a["lord"], "start": fmt_local(a["start"], tz),
                                           "end": fmt_local(a["end"], tz)} for a in md["antar"]],
        "pratyantars_current_ad": ad and [{"lord": x["lord"], "start": fmt_local(x["start"], tz),
                                           "end": fmt_local(x["end"], tz)} for x in ad["pratyantar"]],
    }
    nxt = [p for p in vd["periods"] if md and p["start"] >= md["end"]][:1]
    if nxt:
        chart["vimshottari"]["antardashas_next_md"] = [
            {"lord": a["lord"], "start": fmt_local(a["start"], tz), "end": fmt_local(a["end"], tz)}
            for a in nxt[0]["antar"]]
    chart["transits"] = transits(eph, tr_jd, signs, sav, tz)
    chart["sade_sati"] = sade_sati_periods(eph, jd, signs["Moon"], tz, tr_jd)
    return chart


def years_str(y):
    yrs = int(y)
    rem = (y - yrs) * 12
    mo = int(rem)
    dy = int(round((rem - mo) * 30.4375))
    return f"{yrs}y {mo}m {dy}d"


def panchang_elements(sun_lon, moon_lon):
    diff = norm(moon_lon - sun_lon)
    t = int(diff // 12)  # 0..29
    if t < 15:
        tithi = f"Shukla {TITHIS[t]}" if t < 14 else "Purnima (Full Moon)"
    else:
        tithi = f"Krishna {TITHIS[t - 15]}" if t < 29 else "Amavasya (New Moon)"
    k = int(diff // 6)
    if k == 0:
        karana = "Kimstughna"
    elif k >= 57:
        karana = ["Shakuni", "Chatushpada", "Naga"][k - 57]
    else:
        karana = KARANAS_MOVABLE[(k - 1) % 7]
    yoga = YOGAS_27[int(norm(sun_lon + moon_lon) // NAK_SPAN)]
    nk = nakshatra_info(moon_lon)
    return {"tithi": tithi, "tithi_number": t + 1,
            "tithi_percent_elapsed": round((diff % 12) / 12 * 100, 1),
            "paksha": "Shukla (waxing)" if t < 15 else "Krishna (waning)",
            "nakshatra": f"{nk['name']} pada {nk['pada']}", "yoga": yoga, "karana": karana}


def lord_of(h, lagna_sign):
    return SIGN_LORD[(lagna_sign + h - 1) % 12]


def associated(a, b, signs):
    """Conjunction, mutual (or one-way special) aspect, or sign exchange."""
    if a == b:
        return None
    if signs[a] == signs[b]:
        return "conjunction"
    if SIGN_LORD[signs[a]] == b and SIGN_LORD[signs[b]] == a:
        return "exchange (parivartana)"
    a_asp = any(s == signs[b] for s, _ in aspects_from(a, signs[a]))
    b_asp = any(s == signs[a] for s, _ in aspects_from(b, signs[b]))
    if a_asp and b_asp:
        return "mutual aspect"
    return None


def detect_yogas(chart, pos, signs, lagna_sign):
    Y = []
    moon_s = signs["Moon"]

    def hfm(p):  # house from Moon
        return house_from(signs[p], moon_s)

    def hfl(p):
        return house_from(signs[p], lagna_sign)

    kendras = (1, 4, 7, 10)
    if hfm("Jupiter") in kendras:
        Y.append({"name": "Gajakesari Yoga", "detail": f"Jupiter in house {hfm('Jupiter')} from Moon"})
    if signs["Sun"] == signs["Mercury"]:
        Y.append({"name": "Budhaditya Yoga", "detail": "Sun and Mercury conjunct"
                  + (" (Mercury combust - weaker)" if chart["planets"]["Mercury"]["combust"] else "")})
    if signs["Moon"] == signs["Mars"] or hfm("Mars") == 7:
        Y.append({"name": "Chandra-Mangala Yoga", "detail": "Moon and Mars conjunct/opposed"})
    for p, yname in (("Mars", "Ruchaka"), ("Mercury", "Bhadra"), ("Jupiter", "Hamsa"),
                     ("Venus", "Malavya"), ("Saturn", "Shasha")):
        strong = signs[p] in OWN[p] or signs[p] == EXALT[p][0]
        if strong and hfl(p) in kendras:
            Y.append({"name": f"{yname} Yoga (Pancha Mahapurusha)",
                      "detail": f"{p} {chart['planets'][p]['dignity'].lower()} in house {hfl(p)}"})
    # Moon-based
    others = ["Mars", "Mercury", "Jupiter", "Venus", "Saturn"]
    in2 = [p for p in others if hfm(p) == 2]
    in12 = [p for p in others if hfm(p) == 12]
    if in2 and in12:
        Y.append({"name": "Durudhara Yoga", "detail": f"2nd from Moon: {in2}; 12th: {in12}"})
    elif in2:
        Y.append({"name": "Sunapha Yoga", "detail": f"Planets in 2nd from Moon: {in2}"})
    elif in12:
        Y.append({"name": "Anapha Yoga", "detail": f"Planets in 12th from Moon: {in12}"})
    else:
        canc = [p for p in others if hfl(p) in kendras or hfm(p) in kendras]
        Y.append({"name": "Kemadruma Yoga", "detail": "No planets in 2nd/12th from Moon"
                  + (f"; CANCELLED (Kemadruma bhanga) by {canc} in kendra" if canc else "")})
    ben = [p for p in ("Mercury", "Jupiter", "Venus") if hfm(p) in (6, 7, 8)]
    if len(ben) >= 2:
        Y.append({"name": "Adhi Yoga", "detail": f"Benefics in 6/7/8 from Moon: {ben}"})
    for ref, label in ((lagna_sign, "Lagna"), (moon_s, "Moon")):
        b10 = [p for p in ("Mercury", "Jupiter", "Venus") if house_from(signs[p], ref) == 10]
        if b10:
            Y.append({"name": "Amala Yoga", "detail": f"Benefic {b10} in 10th from {label}"})
    sun_s = signs["Sun"]
    v2 = [p for p in others if house_from(signs[p], sun_s) == 2]
    v12 = [p for p in others if house_from(signs[p], sun_s) == 12]
    if v2 and v12:
        Y.append({"name": "Ubhayachari Yoga", "detail": f"2nd from Sun {v2}, 12th {v12}"})
    elif v2:
        Y.append({"name": "Vesi Yoga", "detail": f"Planets in 2nd from Sun: {v2}"})
    elif v12:
        Y.append({"name": "Vasi Yoga", "detail": f"Planets in 12th from Sun: {v12}"})

    # Yogakaraka & Parashari Raja yogas
    kendra_lords = {h: lord_of(h, lagna_sign) for h in (4, 7, 10)}
    trikona_lords = {h: lord_of(h, lagna_sign) for h in (5, 9)}
    trikona_lords[1] = lord_of(1, lagna_sign)
    for p in SEVEN:
        ks = [h for h, l in kendra_lords.items() if l == p]
        ts = [h for h, l in trikona_lords.items() if l == p and h != 1]
        if ks and ts:
            Y.append({"name": "Yogakaraka", "detail": f"{p} rules kendra {ks} and trikona {ts}"})
    seen = set()
    for kh, kl in list(kendra_lords.items()) + [(1, lord_of(1, lagna_sign))]:
        for th, tl in trikona_lords.items():
            if kl == tl or kh == th:
                continue
            key = tuple(sorted((kl, tl)))
            rel = associated(kl, tl, signs)
            if rel and key not in seen:
                seen.add(key)
                Y.append({"name": "Raja Yoga (Parashari)",
                          "detail": f"Kendra lord {kl} (H{kh}) & trikona lord {tl} (H{th}): {rel}"})
    # Dhana yoga
    wealth = {h: lord_of(h, lagna_sign) for h in (1, 2, 5, 9, 11)}
    seen_d = set()
    for h1, l1 in wealth.items():
        for h2, l2 in wealth.items():
            if h1 >= h2 or l1 == l2:
                continue
            key = tuple(sorted((l1, l2)))
            rel = associated(l1, l2, signs)
            if rel and key not in seen_d:
                seen_d.add(key)
                Y.append({"name": "Dhana Yoga", "detail": f"Lords of H{h1} ({l1}) & H{h2} ({l2}): {rel}"})
    # Viparita raja yoga
    for h, nm in ((6, "Harsha"), (8, "Sarala"), (12, "Vimala")):
        l = lord_of(h, lagna_sign)
        if hfl(l) in (6, 8, 12):
            Y.append({"name": f"{nm} Yoga (Viparita Raja)", "detail": f"H{h} lord {l} in house {hfl(l)}"})
    # Parivartana
    for i, a in enumerate(SEVEN):
        for b in SEVEN[i + 1:]:
            if SIGN_LORD[signs[a]] == b and SIGN_LORD[signs[b]] == a:
                ha, hb = hfl(a), hfl(b)
                kind = "Dainya" if {ha, hb} & {6, 8, 12} else "Khala" if 3 in (ha, hb) else "Maha"
                Y.append({"name": f"{kind} Parivartana Yoga", "detail": f"{a} (H{ha}) <-> {b} (H{hb})"})
    # Neecha bhanga
    for p in SEVEN:
        if signs[p] == (EXALT[p][0] + 6) % 12:
            deb_lord = SIGN_LORD[signs[p]]
            ex_lord = SIGN_LORD[EXALT[p][0]]
            reasons = []
            for l, why in ((deb_lord, "lord of debilitation sign"), (ex_lord, "lord of exaltation sign")):
                if hfl(l) in kendras or hfm(l) in kendras:
                    reasons.append(f"{l} ({why}) in kendra from Lagna/Moon")
            if signs[deb_lord] == signs[p]:
                reasons.append(f"conjunct its dispositor {deb_lord}")
            if reasons:
                Y.append({"name": "Neecha Bhanga Raja Yoga", "detail": f"{p} debilitated, cancelled: " + "; ".join(reasons)})
    return Y


def detect_doshas(pos, signs, lagna_sign):
    D = []
    mars_from = {"Lagna": house_from(signs["Mars"], lagna_sign),
                 "Moon": house_from(signs["Mars"], signs["Moon"]),
                 "Venus": house_from(signs["Mars"], signs["Venus"])}
    hits = {k: v for k, v in mars_from.items() if v in (1, 2, 4, 7, 8, 12)}
    exc = []
    if signs["Mars"] in OWN["Mars"] or signs["Mars"] == EXALT["Mars"][0]:
        exc.append("Mars in own/exaltation sign")
    if signs["Jupiter"] == signs["Mars"] or any(s == signs["Mars"] for s, _ in aspects_from("Jupiter", signs["Jupiter"])):
        exc.append("Jupiter conjoins/aspects Mars")
    D.append({"name": "Mangal (Kuja) Dosha", "present": bool(hits),
              "detail": f"Mars house from Lagna/Moon/Venus: {mars_from}; flagged from {list(hits) or 'none'}",
              "mitigations": exc})
    # Kala Sarpa
    r = pos["Rahu"][0]
    rel = [(pos[p][0] - r) % 360 for p in SEVEN]
    if all(x < 180 for x in rel) or all(x > 180 for x in rel):
        D.append({"name": "Kala Sarpa Yoga", "present": True,
                  "detail": "All seven planets hemmed on one side of the Rahu-Ketu axis"})
    for p in ("Sun", "Moon"):
        for n in ("Rahu", "Ketu"):
            if signs[p] == signs[n]:
                D.append({"name": "Grahana Yoga", "present": True, "detail": f"{p} conjunct {n}"})
    if signs["Jupiter"] == signs["Rahu"]:
        D.append({"name": "Guru-Chandala Yoga", "present": True, "detail": "Jupiter conjunct Rahu"})
    return D


def transits(eph, tr_jd, natal_signs, sav, tz):
    pos = eph.planets(tr_jd)
    out = {"date": fmt_local(tr_jd, tz), "planets": {}}
    for p in PLANETS:
        s = sign_of(pos[p][0])
        out["planets"][p] = {
            "sign": SIGNS[s], "degree": dms(pos[p][0] % 30),
            "retrograde": pos[p][1] < 0 if p not in ("Rahu", "Ketu") else True,
            "house_from_moon": house_from(s, natal_signs["Moon"]),
            "house_from_lagna": house_from(s, natal_signs["Lagna"]),
            "sav_bindus_in_sign": sav[s],
            "nakshatra": nakshatra_info(pos[p][0])["name"],
        }
    sh = out["planets"]["Saturn"]["house_from_moon"]
    out["saturn_status"] = {12: "Sade Sati - rising phase (12th from Moon)",
                            1: "Sade Sati - peak phase (over natal Moon)",
                            2: "Sade Sati - setting phase (2nd from Moon)",
                            4: "Kantaka/Ardhashtama Shani (4th from Moon)",
                            8: "Ashtama Shani (8th from Moon)"}.get(sh, "No Sade Sati / Dhaiya")
    return out


def sade_sati_periods(eph, birth_jd, moon_sign, tz, now_jd, years=100):
    """Scan Saturn's sidereal sign daily across life to list Sade Sati windows."""
    targets = {(moon_sign - 1) % 12, moon_sign, (moon_sign + 1) % 12}
    periods, cur = [], None
    jd = birth_jd
    end = birth_jd + years * 365.25
    while jd <= end:
        xx, _ = swe.calc_ut(jd, swe.SATURN, eph.flags)
        inside = sign_of(xx[0]) in targets
        if inside and cur is None:
            cur = jd
        elif not inside and cur is not None:
            periods.append([cur, jd])
            cur = None
        jd += 1
    if cur is not None:
        periods.append([cur, end])
    merged = []
    for a, b in periods:  # merge brief retrograde exits (< 1 year)
        if merged and a - merged[-1][1] < 365:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [{"start": fmt_local(a, tz), "end": fmt_local(b, tz),
             "status": "current" if a <= now_jd < b else "past" if b < now_jd else "upcoming"}
            for a, b in merged]


# ----------------------------------------------------------------------------
# Matching (Ashtakoota)
# ----------------------------------------------------------------------------
def vashya_of(moon_lon):
    s, d = sign_of(moon_lon), moon_lon % 30
    if s in (0, 1):
        return 0
    if s == 8:
        return 1 if d < 15 else 0
    if s == 9:
        return 0 if d < 15 else 2
    if s in (2, 5, 6, 10):
        return 1
    if s in (3, 11):
        return 2
    if s == 4:
        return 3
    return 4  # Scorpio


def maitri_points(l1, l2):
    if l1 == l2:
        return 5
    a = NATURAL[l1].get(l2, 0)
    b = NATURAL[l2].get(l1, 0)
    table = {(1, 1): 5, (1, 0): 4, (0, 1): 4, (0, 0): 3, (1, -1): 1, (-1, 1): 1,
             (0, -1): 0.5, (-1, 0): 0.5, (-1, -1): 0}
    return table[(a, b)]


def ashtakoota(boy_moon, girl_moon):
    bs, gs = sign_of(boy_moon), sign_of(girl_moon)
    bn, gn = nakshatra_info(boy_moon)["index"], nakshatra_info(girl_moon)["index"]
    res = []
    v = 1 if VARNA_RANK[bs] >= VARNA_RANK[gs] else 0
    res.append(("Varna", 1, v, f"{VARNA_NAME[VARNA_RANK[bs]]} / {VARNA_NAME[VARNA_RANK[gs]]}", "work/ego compatibility"))
    bv, gv = vashya_of(boy_moon), vashya_of(girl_moon)
    res.append(("Vashya", 2, VASHYA_TABLE[bv][gv], f"{VASHYA_NAMES[bv]} / {VASHYA_NAMES[gv]}", "mutual attraction, influence"))
    t1 = ((bn - gn) % 27 + 1) % 9  # girl -> boy
    t2 = ((gn - bn) % 27 + 1) % 9  # boy -> girl
    tara = (0 if t1 in (3, 5, 7) else 1.5) + (0 if t2 in (3, 5, 7) else 1.5)
    res.append(("Tara/Dina", 3, tara, f"counts {t1 or 9} & {t2 or 9} (3,5,7 inauspicious)", "health & well-being"))
    by, gy = NAK_YONI[bn], NAK_YONI[gn]
    res.append(("Yoni", 4, YONI_TABLE[by][gy], f"{YONI_ANIMALS[by]} / {YONI_ANIMALS[gy]}", "physical & sexual compatibility"))
    bl, gl = SIGN_LORD[bs], SIGN_LORD[gs]
    res.append(("Graha Maitri", 5, maitri_points(bl, gl), f"{bl} / {gl}", "mental compatibility, friendship"))
    bg, gg = NAK_GANA[bn], NAK_GANA[gn]
    res.append(("Gana", 6, GANA_TABLE[bg][gg], f"{GANA_NAMES[bg]} / {GANA_NAMES[gg]}", "temperament"))
    dist = (gs - bs) % 12 + 1
    bad = dist in (2, 12, 5, 9, 6, 8)
    res.append(("Bhakoot", 7, 0 if bad else 7, f"{SIGNS[bs]} / {SIGNS[gs]} ({dist}/{(bs - gs) % 12 + 1})", "family welfare, finances, love"))
    bnd, gnd = NADI_CYCLE[bn % 6], NADI_CYCLE[gn % 6]
    res.append(("Nadi", 8, 0 if bnd == gnd else 8, f"{NADI_NAMES[bnd]} / {NADI_NAMES[gnd]}", "health, genetic compatibility, progeny"))
    total = sum(r[2] for r in res)
    notes = []
    if bnd == gnd:
        if bn == gn and nakshatra_info(boy_moon)["pada"] != nakshatra_info(girl_moon)["pada"]:
            notes.append("Nadi dosha mitigated: same nakshatra but different padas.")
        elif bs == gs and bn != gn:
            notes.append("Nadi dosha mitigated: same Moon sign, different nakshatras.")
        elif bn == gn and bs != gs:
            notes.append("Nadi dosha mitigated: same nakshatra, different Moon signs.")
        else:
            notes.append("Nadi dosha present.")
    if bad:
        if bl == gl or maitri_points(bl, gl) >= 4:
            notes.append("Bhakoot dosha mitigated: Moon-sign lords are the same or friendly.")
        else:
            notes.append("Bhakoot dosha present.")
    if GANA_TABLE[bg][gg] <= 1 and bl == gl:
        notes.append("Gana dosha mitigated: same Moon-sign lord.")
    verdict = ("Excellent (>= 28)" if total >= 28 else "Very good (24-27.5)" if total >= 24
               else "Acceptable (18-23.5)" if total >= 18 else "Below traditional minimum of 18")
    return {"kootas": [{"koota": r[0], "max": r[1], "points": r[2], "detail": r[3], "signifies": r[4]} for r in res],
            "total": total, "out_of": 36, "verdict": verdict, "notes": notes}


# ----------------------------------------------------------------------------
# Daily panchang
# ----------------------------------------------------------------------------
def daily_panchang(eph, date_s, tz, lat, lon):
    local_midnight = datetime.strptime(date_s, "%Y-%m-%d").replace(tzinfo=tz)
    jd0 = jd_from_utc(local_midnight.astimezone(timezone.utc))
    sr = eph.sun_event(jd0, lat, lon, True)
    ss = eph.sun_event(sr, lat, lon, False)
    nsr = eph.sun_event(ss, lat, lon, True)
    pos = eph.planets(sr)
    el = panchang_elements(pos["Sun"][0], pos["Moon"][0])
    wd = local_midnight.weekday()
    day = (ss - sr) / 8
    sun_wd = (wd + 1) % 7  # Sunday = 0
    rahu = [8, 2, 7, 5, 6, 4, 3][sun_wd]
    yama = [5, 4, 3, 2, 1, 7, 6][sun_wd]
    guli = [7, 6, 5, 4, 3, 2, 1][sun_wd]

    def seg(i):
        a = sr + (i - 1) * day
        return f"{fmt_local(a, tz, True)[11:]} - {fmt_local(a + day, tz, True)[11:]}"

    mid = sr + (ss - sr) / 2
    muh = (ss - sr) / 15
    # end times of the current tithi / nakshatra (search forward minute-steps via bisection)
    def next_change(fn, start):
        v0 = fn(start)
        step = 1 / 24
        t = start
        while fn(t + step) == v0 and t - start < 3:
            t += step
        lo, hi = t, t + step
        for _ in range(30):
            m = (lo + hi) / 2
            if fn(m) == v0:
                lo = m
            else:
                hi = m
        return hi

    def tithi_idx(t):
        p = eph.planets(t)
        return int(norm(p["Moon"][0] - p["Sun"][0]) // 12)

    def nak_idx(t):
        return int(eph.planets(t)["Moon"][0] // NAK_SPAN)

    out = {"date": date_s, "vara": VARA[wd], "sunrise": fmt_local(sr, tz, True)[11:],
           "sunset": fmt_local(ss, tz, True)[11:], "next_sunrise": fmt_local(nsr, tz, True),
           **{f"{k}_at_sunrise": v for k, v in el.items()},
           "tithi_ends": fmt_local(next_change(tithi_idx, sr), tz, True),
           "nakshatra_ends": fmt_local(next_change(nak_idx, sr), tz, True),
           "moon_sign_at_sunrise": SIGNS[sign_of(pos["Moon"][0])],
           "sun_sign": SIGNS[sign_of(pos["Sun"][0])],
           "rahu_kaal": seg(rahu), "yamaganda": seg(yama), "gulika_kaal": seg(guli),
           "abhijit_muhurta": (f"{fmt_local(mid - muh / 2, tz, True)[11:]} - {fmt_local(mid + muh / 2, tz, True)[11:]}"
                               + (" (traditionally avoided on Wednesday)" if wd == 2 else "")),
           "ayanamsa": f"{eph.ayanamsa_name} {dms(eph.ayanamsa(sr))}"}
    return out


# ----------------------------------------------------------------------------
# Text rendering
# ----------------------------------------------------------------------------
SOUTH_GRID = [[11, 0, 1, 2], [10, None, None, 3], [9, None, None, 4], [8, 7, 6, 5]]


def south_indian(title, occupants, lagna_sign, w=13):
    def cell(s):
        items = (["As"] if s == lagna_sign else []) + list(occupants.get(s, []))
        txt = " ".join(items)
        return [SIGN_ABBR[s].ljust(w), txt[:w].ljust(w), txt[w:2 * w].ljust(w)]

    full = "+" + "+".join(["-" * w] * 4) + "+"
    side = "+" + "-" * w + "+" + " " * (2 * w + 1) + "+" + "-" * w + "+"
    lines = [full]
    for r, row in enumerate(SOUTH_GRID):
        cells = [cell(s) if s is not None else None for s in row]
        for k in range(3):
            if cells[1] is None:
                mid = title.center(2 * w + 1) if (r == 1 and k == 2) else " " * (2 * w + 1)
                lines.append("|" + cells[0][k] + "|" + mid + "|" + cells[3][k] + "|")
            else:
                lines.append("|" + "|".join(c[k] for c in cells) + "|")
        lines.append(side if r in (1,) else full)
    return "\n".join(lines)


def render_chart(c, name=None):
    o = []
    P = c["planets"]
    L = c["lagna"]
    o.append("=" * 78)
    o.append(f"VEDIC BIRTH CHART (JANMA KUNDALI){' - ' + name if name else ''}")
    o.append("=" * 78)
    i, s = c["input"], c["settings"]
    o.append(f"Birth: {i['local_time']}  (UTC {i['utc']})  Lat {i['latitude']}  Lon {i['longitude']}")
    o.append(f"Ayanamsa: {s['ayanamsa']} {s['ayanamsa_value']} | Nodes: {s['node']} | Houses: {s['houses']}")
    o.append(f"Ephemeris: {s['ephemeris']}")
    o.append("")
    o.append(f"LAGNA: {L['sign']} {L['degree_in_sign']}  | {L['nakshatra']['name']} pada {L['nakshatra']['pada']} "
             f"| Lagna lord: {L['lord']} | Navamsa lagna: {L['vargas']['D9']}")
    ts = c["time_sensitivity"]
    o.append(f"Birth-time sensitivity: Lagna changes {ts['lagna_changes_in_minutes']['earlier']} min earlier / "
             f"{ts['lagna_changes_in_minutes']['later']} min later; D9 lagna changes "
             f"{ts['navamsa_lagna_changes_in_minutes']['earlier']}/{ts['navamsa_lagna_changes_in_minutes']['later']} min")
    o.append("")
    o.append(f"{'Graha':8} {'Sign':12} {'Degree':11} {'H':>2} {'Chl':>3} {'Nakshatra':18} {'Pd':>2} {'NkLd':8} {'D9':12} Status")
    o.append("-" * 110)
    for p in PLANETS:
        d = P[p]
        flags = []
        if d["retrograde"] and p not in ("Rahu", "Ketu"):
            flags.append("R")
        if d["combust"]:
            flags.append("Combust")
        o.append(f"{p:8} {d['sign']:12} {d['degree_in_sign']:11} {d['house']:>2} {d['bhava_chalit_house']:>3} "
                 f"{d['nakshatra']['name']:18} {d['nakshatra']['pada']:>2} {d['nakshatra']['lord']:8} "
                 f"{d['vargas']['D9']:12} {d['dignity']}{' [' + ', '.join(flags) + ']' if flags else ''}")
    o.append("")
    occ = {}
    occ9 = {}
    occ10 = {}
    for p in PLANETS:
        occ.setdefault(P[p]["sign_index"], []).append(ABBR[p] + ("(R)" if P[p]["retrograde"] and p not in ("Rahu", "Ketu") else ""))
        occ9.setdefault(SIGNS.index(P[p]["vargas"]["D9"]), []).append(ABBR[p])
        occ10.setdefault(SIGNS.index(P[p]["vargas"]["D10"]), []).append(ABBR[p])
    o.append(south_indian("RASHI (D1)", occ, L["sign_index"]))
    o.append("")
    o.append(south_indian("NAVAMSA (D9)", occ9, SIGNS.index(L["vargas"]["D9"])))
    o.append("")
    o.append(south_indian("DASAMSA (D10)", occ10, SIGNS.index(L["vargas"]["D10"])))
    o.append("(South Indian style: signs fixed, Pisces top-left, Aries next, clockwise.)")
    o.append("")
    o.append("HOUSES")
    for h in c["houses"]:
        o.append(f" H{h['house']:<2} {h['sign']:12} lord {h['lord']:8} -> H{h['lord_placed_in_house']:<2} "
                 f"occupants: {', '.join(h['occupants']) or '-':24} aspected by: {', '.join(h['aspected_by']) or '-'}")
    o.append("")
    o.append("DIVISIONAL CHARTS (sign of each graha)")
    hdr = "Varga ".ljust(8) + "Asc".ljust(5) + "".join(ABBR[p].ljust(5) for p in PLANETS)
    o.append(hdr)
    for n in VARGAS:
        row = f"D{n}".ljust(8) + SIGN_ABBR[SIGNS.index(L["vargas"][f"D{n}"])].ljust(5)
        row += "".join(SIGN_ABBR[SIGNS.index(P[p]["vargas"][f"D{n}"])].ljust(5) for p in PLANETS)
        o.append(row + "  " + VARGAS[n])
    o.append("")
    pc = c["panchang"]
    o.append(f"PANCHANG AT BIRTH: {pc['vara']} | Tithi {pc['tithi']} ({pc['paksha']}) | Nakshatra {pc['nakshatra']} | "
             f"Yoga {pc['yoga']} | Karana {pc['karana']}")
    o.append("")
    o.append("JAIMINI: " + ", ".join(f"{k} {v}" for k, v in c["karakas"].items()))
    o.append("Arudhas: " + ", ".join(f"{k} {v['sign']} (H{v['house']})" for k, v in c["arudhas"].items()))
    o.append("")
    o.append("ASHTAKAVARGA (SAV by house; 28+ strong, <25 weak; total 337)")
    o.append("  " + "  ".join(f"H{h}:{v}" for h, v in c["ashtakavarga"]["sav_by_house"].items()))
    o.append("  BAV by sign: " + "; ".join(f"{p[:2]} " + "".join(str(x) for x in v.values())
                                           for p, v in c["ashtakavarga"]["bav"].items()) + "  (Ar..Pi)")
    o.append("")
    o.append("YOGAS DETECTED")
    for y in c["yogas"]:
        o.append(f" - {y['name']}: {y['detail']}")
    o.append("DOSHAS / NOTABLE COMBINATIONS")
    for d in c["doshas"]:
        o.append(f" - {d['name']}: {'present' if d['present'] else 'absent'} - {d['detail']}"
                 + (f" | mitigations: {d['mitigations']}" if d.get("mitigations") else ""))
    o.append("")
    v = c["vimshottari"]
    o.append(f"VIMSHOTTARI DASHA (balance at birth: {v['balance_at_birth']})")
    o.append("  " + " | ".join(f"{m['lord']} {m['start'][:7]}->{m['end'][:7]}" for m in v["mahadashas"]))
    cur = v["current"]
    if cur["mahadasha"]:
        o.append(f"  As of {v['as_of']}: {cur['mahadasha']['lord']} MD / {cur['antardasha']['lord']} AD / "
                 f"{cur['pratyantar']['lord']} PD (PD ends {cur['pratyantar']['end']})")
        o.append("  Antardashas in current MD: " + " | ".join(f"{a['lord']} {a['start']}" for a in v["antardashas_current_md"]))
        o.append("  Pratyantars in current AD: " + " | ".join(f"{a['lord']} {a['start']}" for a in v["pratyantars_current_ad"]))
    o.append("")
    t = c["transits"]
    o.append(f"GOCHARA (TRANSITS) on {t['date']}: {t['saturn_status']}")
    for p, d in t["planets"].items():
        o.append(f"  {p:8} {d['sign']:12} {d['degree']}{' R' if d['retrograde'] and p not in ('Rahu', 'Ketu') else '  '}  "
                 f"H{d['house_from_moon']:<2} from Moon, H{d['house_from_lagna']:<2} from Lagna, SAV {d['sav_bindus_in_sign']}")
    o.append("SADE SATI windows: " + "; ".join(f"{p['start']} to {p['end']} ({p['status']})" for p in c["sade_sati"]))
    return "\n".join(o)


def render_match(m, b_info, g_info):
    o = ["ASHTAKOOTA GUNA MILAN", "=" * 60,
         f"Boy : Moon {b_info['sign']} | {b_info['nak']}",
         f"Girl: Moon {g_info['sign']} | {g_info['nak']}", "",
         f"{'Koota':14} {'Pts':>5} / {'Max':<3}  Detail"]
    for k in m["kootas"]:
        o.append(f"{k['koota']:14} {k['points']:>5} / {k['max']:<3}  {k['detail']}  [{k['signifies']}]")
    o.append(f"\nTOTAL: {m['total']} / 36  -> {m['verdict']}")
    for n in m["notes"]:
        o.append(f" * {n}")
    o.append(f"\nMangal dosha - Boy: {b_info['manglik']} | Girl: {g_info['manglik']}")
    return "\n".join(o)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Vedic astrology calculator (Swiss Ephemeris)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--ayanamsa", default="lahiri", choices=list(AYANAMSAS))
        p.add_argument("--node", default="mean", choices=["mean", "true"])
        p.add_argument("--json", action="store_true")

    c = sub.add_parser("chart")
    c.add_argument("--date", required=True, help="YYYY-MM-DD (local birth date)")
    c.add_argument("--time", required=True, help="HH:MM[:SS] local birth time (24h)")
    c.add_argument("--tz", required=True, help="IANA zone (Asia/Kolkata) or offset (+05:30)")
    c.add_argument("--lat", type=float, required=True, help="latitude, north positive")
    c.add_argument("--lon", type=float, required=True, help="longitude, east positive")
    c.add_argument("--name")
    c.add_argument("--on", help="transit/dasha reference date YYYY-MM-DD (default: today)")
    c.add_argument("--year-days", type=float, default=365.25, help="dasha year length (365.25 / 365.2422 / 360)")
    common(c)

    m = sub.add_parser("match")
    m.add_argument("--boy", required=True, help="'YYYY-MM-DD,HH:MM,TZ,lat,lon'")
    m.add_argument("--girl", required=True, help="'YYYY-MM-DD,HH:MM,TZ,lat,lon'")
    common(m)

    p = sub.add_parser("panchang")
    p.add_argument("--date", required=True)
    p.add_argument("--tz", required=True)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    common(p)

    a = ap.parse_args()
    eph = Ephemeris(a.ayanamsa, a.node)

    if a.cmd == "chart":
        tz = parse_tz(a.tz)
        local, utc = local_to_utc(a.date, a.time, tz)
        tr = None
        if a.on:
            tr = datetime.strptime(a.on, "%Y-%m-%d").replace(hour=12, tzinfo=tz).astimezone(timezone.utc)
        chart = build_chart(eph, local, utc, a.lat, a.lon, tr, a.year_days)
        print(json.dumps(chart, indent=1, default=str) if a.json else render_chart(chart, a.name))

    elif a.cmd == "match":
        infos, moons = [], []
        for spec in (a.boy, a.girl):
            d, t, z, la, lo = [x.strip() for x in spec.split(",")]
            tz = parse_tz(z)
            local, utc = local_to_utc(d, t, tz)
            jd = jd_from_utc(utc)
            pos = eph.planets(jd)
            asc, _, _ = eph.lagna(jd, float(la), float(lo))
            signs = {pp: sign_of(l) for pp, (l, _) in pos.items()}
            dos = detect_doshas(pos, signs, sign_of(asc))[0]
            nk = nakshatra_info(pos["Moon"][0])
            moons.append(pos["Moon"][0])
            infos.append({"sign": SIGNS[signs["Moon"]], "nak": f"{nk['name']} pada {nk['pada']}",
                          "manglik": ("YES " if dos["present"] else "no ") + dos["detail"]
                          + (f" | mitigations: {dos['mitigations']}" if dos["mitigations"] else "")})
        res = ashtakoota(moons[0], moons[1])
        res["boy"], res["girl"] = infos
        print(json.dumps(res, indent=1) if a.json else render_match(res, infos[0], infos[1]))

    elif a.cmd == "panchang":
        tz = parse_tz(a.tz)
        res = daily_panchang(eph, a.date, tz, a.lat, a.lon)
        if a.json:
            print(json.dumps(res, indent=1))
        else:
            for k, v in res.items():
                print(f"{k.replace('_', ' ').title():28} {v}")


if __name__ == "__main__":
    main()
