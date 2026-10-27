# Timing: Dashas, Transits, Sade Sati, Muhurta

## Vimshottari Dasha (120-year cycle)

Order and years: Ketu 7 → Venus 20 → Sun 6 → Moon 10 → Mars 7 → Rahu 18 →
Jupiter 16 → Saturn 19 → Mercury 17. The starting period comes from the
Moon's nakshatra lord. The balance depends on how far the Moon has travelled
through that nakshatra. The script computes MD (Mahadasha), AD (Antardasha /
Bhukti) and PD (Pratyantar) dates.

### Reading a dasha lord

A dasha lord gives the results of:
1. **The houses it rules** (from the Lagna, and also from the Moon).
2. **The house it occupies**, and the planets conjunct or aspecting it.
3. **Its dignity and strength** in D1 and D9 (and D10 for career matters).
4. **Its nakshatra lord.** The planet delivers what its star lord promises.
   This is often the most predictive layer.
5. **Natural karakatva.** Jupiter's dasha brings children, teachers and
   learning. Venus's brings relationships, comforts and the arts.
6. **Its yogas.** Raja and dhana yogas fructify in the dashas of the planets
   forming them.

Rahu and Ketu dashas act through their sign lord (dispositor), their
nakshatra lord and the planets they conjoin.

### MD–AD interplay

- Read the AD lord's placement **from the MD lord**. In 1/5/9 or 1/4/7/10
  from it, results flow well. In 6/8/12 from it (shadashtaka/dwirdwadasha),
  expect friction, obstacles or health matters, even if both are benefic.
- If the MD and AD lords are natural friends and functionally benefic for the
  Lagna, the period is good. If both are functional malefics or marakas, take
  care (but don't predict death).
- The first AD of every MD (the MD lord's own sub-period) shows the theme of
  the whole MD.
- **Dasha chhidra** (the last AD of an MD) is often a transition or loss of
  the old. The **dasha sandhi** (junction between two MDs) is a period of
  change.

### Event-timing recipe

1. **Promise:** does the natal chart (D1 + varga) show the event? Use the
   triple check from SKILL.md.
2. **Dasha:** is the MD/AD lord connected to the house, its lord, or the
   karaka (ownership, placement, aspect, conjunction, nakshatra)?
3. **Double transit (K.N. Rao's method):** both Jupiter and Saturn aspect
   or transit the relevant house or its lord (counted from the Lagna, and
   also from the Moon). This is reliable for marriage, childbirth, career
   changes and property.
4. **Fine-tuning:** a Mars or Sun transit over sensitive points gives the
   month. Then the PD.

Typical significators:
- **Marriage:** the 7th house and lord, Venus (and Jupiter for a woman's
  husband), the D9 lagna and 7th, the Upapada lagna, and the dashas of
  Rahu or of planets in or aspecting the 7th.
- **Career start or change:** the 10th lord, planets in the 10th, the D10
  lagna lord, the Amatyakaraka, and the 6th (service) and 11th (gains)
  houses.
- **Children:** the 5th house and lord, Jupiter, the D7 chart, and the
  Putrakaraka.
- **Property:** the 4th house and lord, Mars (land), Venus (vehicles), and D4.
- **Foreign settlement:** the 12th, 9th, 7th and 4th (leaving home), Rahu,
  and movable signs.
- **Wealth:** the 2nd, 11th, 5th and 9th lords, dhana yogas, and SAV of
  the 11th.

### Alternate dashas (advanced; mention if the user asks)
- **Yogini** (36-year cycle): a good confirmer.
- **Chara dasha** (Jaimini): sign-based, strong for life events. Read with
  the arudhas and karakas.
- **Ashtottari**: used in some traditions when Rahu is in a kendra or
  trikona from the Lagna lord.
- **Kalachakra**: complex. Use only if the user explicitly asks.
The script calculates Vimshottari only, so state that when discussing the others.

## Gochara (Transits)

Transits are judged primarily **from the natal Moon** (Chandra Lagna), and
secondarily from the Lagna. Classical favourable houses from the Moon:

| Planet | Good houses from Moon | Vedha (obstruction) house |
|---|---|---|
| Sun | 3, 6, 10, 11 | 9, 12, 4, 5 |
| Moon | 1, 3, 6, 7, 10, 11 | 5, 9, 12, 2, 4, 8 |
| Mars | 3, 6, 11 | 12, 9, 5 |
| Mercury | 2, 4, 6, 8, 10, 11 | 5, 3, 9, 1, 8, 12 |
| Jupiter | 2, 5, 7, 9, 11 | 12, 4, 3, 10, 8 |
| Venus | 1, 2, 3, 4, 5, 8, 9, 11, 12 | 8, 7, 1, 10, 9, 5, 11, 3, 6 |
| Saturn | 3, 6, 11 | 12, 9, 5 |
| Rahu/Ketu | 3, 6, 11 | 12, 9, 5 |

(Vedha: a good transit is blocked if another planet, except a father–son
pair such as Sun–Saturn or Moon–Mercury, simultaneously transits the
corresponding vedha house. Mention vedha only for detailed transit work.)

**Refine with Ashtakavarga:** a transit through a sign where the planet's
own BAV has ≥ 4 bindus (and SAV ≥ 28) gives good results even in a
"bad" house. A transit through a sign with 0–3 BAV bindus (SAV < 25) is
weaker or harder. The script lists the SAV of each transiting planet's
sign. The BAV is in the JSON.

Major transits to always comment on for a "current period" question:
- **Saturn** (~2.5 years per sign): Sade Sati, Ashtama Shani (8th from the
  Moon: hard, health and obstacles), Kantaka/Ardhashtama (4th: domestic
  pressure). Transits of the 3rd, 6th and 11th from the Moon are excellent.
- **Jupiter** (~1 year per sign): the 2nd, 5th, 7th, 9th and 11th from the
  Moon bring growth and opportunities. Its aspects (5, 7, 9) bless houses.
- **Rahu/Ketu axis** (~18 months per sign): obsession and change on the
  Rahu house, release on the Ketu house. Watch eclipses near natal
  luminaries or the Lagna.

## Sade Sati (7½ years of Saturn)

Saturn transits the 12th, 1st and 2nd from the natal Moon, about 2.5 years
each. The script lists every window in a lifetime.
- **Rising phase (12th):** expenses, sleep, letting go, travel, mental strain.
- **Peak (over the Moon):** emotional and mental pressure, health,
  restructuring of identity, hard lessons, maturity.
- **Setting phase (2nd):** finances, family, speech. Consolidation.

Present it honestly and constructively:
- Intensity depends on Saturn's natal functional role (it is mild or even
  rewarding for Taurus, Libra, Capricorn and Aquarius lagnas and Moon
  signs), its natal strength, the SAV bindus in the transited signs, and
  the running dasha.
- Many people achieve major success during Sade Sati, because Saturn
  rewards discipline, service and honesty.
- Most people go through 2–3 Sade Satis in a lifetime. The first is often
  the hardest in youth. The second, in mid-life, brings career and family
  responsibility.
- Remedies: see `remedies.md` (Shani mantra, Hanuman Chalisa, Saturday
  service and charity, discipline).

## Panchang & Muhurta essentials

The five limbs: **Tithi** (lunar day), **Vara** (weekday, sunrise to sunrise),
**Nakshatra**, **Yoga** (Sun + Moon), **Karana** (half-tithi).

- **Tithi groups:** Nanda (1, 6, 11): joy. Bhadra (2, 7, 12): good for
  health and stable works. Jaya (3, 8, 13): victory. Rikta (4, 9, 14):
  empty, avoid for auspicious starts. Purna (5, 10, 15): fullness.
  Amavasya suits ancestral rites, not new beginnings.
- **Avoid for auspicious starts:** Vishti (Bhadra) karana, Rahu Kaal,
  Yamaganda, Gulika (for most activities), inauspicious yogas (Vyatipata,
  Vaidhriti, Vishkambha, Atiganda, Shula, Ganda, Vyaghata, Vajra, Parigha),
  and eclipses.
- **Nakshatras by activity:**
  - Fixed (Rohini, Uttara Phalguni/Ashadha/Bhadrapada): foundations, housewarming.
  - Movable (Swati, Punarvasu, Shravana, Dhanishta, Shatabhisha): travel, vehicles.
  - Light/swift (Ashwini, Pushya, Hasta): medicine, trade, learning.
  - Soft (Mrigashira, Chitra, Anuradha, Revati): arts, friendship, marriage.
  - Sharp/harsh (Ardra, Ashlesha, Jyeshtha, Mula): surgery, dealing with enemies.
  - Fierce (Bharani, Magha, Purva trio): strong or decisive actions.
- **Tara bala** (personal suitability of the day's Moon nakshatra):
  count from the janma nakshatra to the day's nakshatra, then take the
  remainder mod 9 (0 → 9). 1 Janma (mixed), 2 Sampat (wealth ✓),
  3 Vipat (danger ✗), 4 Kshema (well-being ✓), 5 Pratyak (obstacles ✗),
  6 Sadhana (achievement ✓), 7 Naidhana (destruction ✗), 8 Mitra (friend ✓),
  9 Parama Mitra (great friend ✓).
- **Chandra bala:** the day's transiting Moon should be in 1, 3, 6, 7, 10
  or 11 from the natal Moon.
- **Marriage muhurta** additionally needs Jupiter and Venus not combust,
  avoids Kharmas (Sun in Sagittarius or Pisces) and Chaturmas per regional
  custom, and suits a strong lagna for the ceremony time.
- **Abhijit muhurta** (around local noon) is broadly auspicious and
  overrides many defects, except on Wednesdays.

For a muhurta request, scan candidate dates with the `panchang`
subcommand (loop over dates) and present the 2–3 best options with
reasons. Note that a family priest's regional panchang may differ slightly
(sunrise definition, ayanamsa, Amanta vs Purnimanta month naming).
