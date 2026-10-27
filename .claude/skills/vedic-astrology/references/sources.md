# Sources & Benchmarks

## Calculation standards (what leading software does, and what this skill does)

| Aspect | Industry practice (Jagannatha Hora, Parashara's Light, Kala, AstroSage, Drik Panchang) | This skill |
|---|---|---|
| Ephemeris | Swiss Ephemeris (JPL-derived) | Swiss Ephemeris via pyswisseph (Moshier built-in; set `SE_EPHE_PATH` for the full files) |
| Zodiac | Sidereal | Sidereal |
| Ayanamsa | Lahiri/Chitrapaksha (Indian Govt. standard, Rashtriya Panchang). KP for Krishnamurti users. Raman for followers of B.V. Raman | Lahiri default. Raman, KP, True Chitra, Yukteshwar selectable |
| Houses | Whole sign (Rashi chart) + Bhava Chalit (Sripati) | Both |
| Nodes | Mean (most traditional) or True | Mean default, True selectable |
| Dasha year | 365.25 days (Savana 360 / sidereal 365.256 alternatives) | 365.25 default, configurable |
| Vargas | 16 Shodasavargas per BPHS | All 16 (D1–D60) |
| Ashtakavarga | BPHS tables, SAV 337 | Same, checked by an assertion |
| Matching | Ashtakoota 36 gunas with parihara | Same + cancellation notes |

Small differences against other software (a few arc-minutes, a dasha date
a few days apart) usually come from ayanamsa variant, true vs mean node,
year length, or a time-zone/LMT entry for old births. Check those first
if the user compares results.

## Classical texts (cite when helpful)

- **Brihat Parashara Hora Shastra (BPHS)**, Maharishi Parashara: the
  foundation (grahas, bhavas, vargas, dashas, ashtakavarga, yogas, remedies).
- **Brihat Jataka**, Varahamihira (6th c.): concise classic of natal astrology.
- **Saravali**, Kalyana Varma: detailed planetary combinations.
- **Phaladeepika**, Mantreswara: practical predictive rules, yogas, gochara.
- **Jataka Parijata**, Vaidyanatha Dikshita: yogas and longevity.
- **Uttara Kalamrita**, Kalidasa: house significations, karakatvas.
- **Jaimini Sutras**: chara karakas, arudhas, chara dasha.
- **Prasna Marga**: horary, remedies, ethics of the astrologer.
- **Muhurta Chintamani**: electional astrology.

## Modern authorities

- **B.V. Raman**: *How to Judge a Horoscope*, *300 Important Combinations*,
  *Hindu Predictive Astrology*.
- **K.N. Rao**: dasha and the double-transit method, Chara dasha
  popularisation, rigorous research.
- **P.V.R. Narasimha Rao**: *Vedic Astrology: An Integrated Approach*;
  author of Jagannatha Hora.
- **Sanjay Rath**: Jaimini and Parashari integration (*Crux of Vedic Astrology*).
- **K.S. Krishnamurti**: the KP system (sub-lords, Placidus cusps). Use
  `--ayanamsa kp` if the user follows KP. Note that KP interpretation uses
  Placidus cusps and sub-lords, which this skill covers only partially.
