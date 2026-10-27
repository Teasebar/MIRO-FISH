# Kundli Milan (Marriage Compatibility)

## 1. Ashtakoota Guna Milan (computed by `match`)

| Koota | Max | Based on | Signifies |
|---|---|---|---|
| Varna | 1 | Moon-sign varna (Brahmin: water signs; Kshatriya: fire; Vaishya: earth; Shudra: air) | Spiritual or ego compatibility; groom's varna ≥ bride's |
| Vashya | 2 | Moon-sign category (quadruped, human, water, wild, insect) | Mutual attraction and influence |
| Tara (Dina) | 3 | Nakshatra count both ways, mod 9 | Health, well-being, destiny together |
| Yoni | 4 | Nakshatra animal | Physical and sexual harmony, instinctive comfort |
| Graha Maitri | 5 | Friendship of the Moon-sign lords | Mental wavelength, friendship. **Most important for day-to-day harmony** |
| Gana | 6 | Deva / Manushya / Rakshasa temperament | Temperament, behaviour |
| Bhakoot | 7 | Relative Moon-sign positions (2/12, 5/9 and 6/8 are bad) | Family welfare, finances, love |
| Nadi | 8 | Adi / Madhya / Antya | Physiological compatibility, progeny, health |

**Thresholds** (traditional): < 18 not recommended · 18–24 acceptable ·
24–32 very good · 32+ excellent. A high score with serious chart issues is
worse than a moderate score with strong charts.

North Indian practice is Ashtakoota. South Indian practice uses **Dasha
Porutham** (10 poruthams). If the user is from South India, mention it
and note that the key items overlap (Dina, Gana, Yoni, Rasi, Rasyadhipati,
Rajju, Vedha, Vasya, Mahendra, Stree Deergha). Rajju (the same "body part"
group of nakshatras) and Vedha (mutually obstructing nakshatra pairs) are
considered critical there.

## 2. Dosha cancellations (Parihara)

- **Nadi dosha (0/8)** is cancelled or greatly reduced when: both have the
  same nakshatra but different padas; the same Moon sign but different
  nakshatras; the same nakshatra but different Moon signs; the Moon-sign
  lords are the same or friends (per some texts); or Graha Maitri and
  Bhakoot are both full. The script notes the first three.
- **Bhakoot dosha (0/7)** is cancelled when the Moon-sign lords are the
  same (for example Aries–Scorpio 6/8, both Mars) or mutual friends, or
  when the Nadi and Tara scores are good.
- **Gana dosha** is reduced if the Moon-sign lords are the same or friendly,
  or if the Yoni and Bhakoot are good.

Always state the cancellation logic rather than just "dosha present".

## 3. Beyond gunas — the real compatibility analysis

The guna system uses only the two Moons. Skilled astrologers always also check:

1. **The 7th house of each chart:** its lord's strength, occupants and
   aspects. Malefics there without support show friction.
2. **Venus (both charts) and Jupiter (the woman's chart, traditionally):**
   their strength and affliction.
3. **Navamsa (D9) of each:** the D9 lagna lord and 7th lord, and the
   condition of Venus and Jupiter there. This is the chart of marriage.
4. **Mangal dosha balance:** compare the two charts (see
   `yogas-doshas.md`). Both Manglik, or one Manglik with strong
   cancellations, is acceptable.
5. **Cross-chart overlays:** one partner's Moon or Lagna falling in the
   other's 1/5/7/9/11 is harmonious. In 6/8/12 it is tense. Jupiter or
   Venus of one on the other's Moon or Lagna is very favourable. Saturn
   of one on the other's Moon is heavy but committed.
6. **Dasha sync:** look at the dashas both will run in the first 5–10
   years of marriage. Avoid both entering harsh 6/8/12-lord or maraka
   periods together. A 7th-lord or Venus period for one or both soon after
   the wedding is favourable.
7. **Longevity of the marriage:** the 2nd (family) and 8th (marital
   longevity, mangalya sthana in a woman's chart) houses free of heavy
   affliction.
8. **Upapada lagna (UL)** and its 2nd house: Jaimini's indicator of
   marriage sustenance.

## 4. Presenting a match

```
## Compatibility: A & B
Guna score: 26/36 (very good), table with one-line meanings
Doshas & cancellations
Chart-level analysis: 7th houses, Venus/Jupiter, D9, Mangal balance, overlays, dasha sync
Strengths of this union · Areas needing conscious effort
Overall assessment (nuanced; never "doomed" or "perfect") + practical advice
```

Be sensitive. Families may make real decisions on this. Emphasise that
compatibility is ultimately built through communication, values and
effort, and that astrology highlights areas to work on. If the user is
choosing between suitors, compare fairly and avoid a deterministic verdict.
For same-sex couples or users who don't fit the boy/girl roles, run the
calculation both ways (roles affect only Varna, Vashya, Gana and Tara
direction) and present both, or the more favourable one with a note.
