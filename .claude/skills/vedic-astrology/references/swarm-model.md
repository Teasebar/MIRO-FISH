# Jyotish Swarm: MiroFish-style prediction without API credits

`scripts/swarm_predict.py` reuses the **prediction architecture of MiroFish**
(a multi-agent "swarm intelligence" engine) but replaces its LLM-driven
agents with deterministic Jyotish rules. It makes **no API calls**:
one run of 1,000 simulations over 24 months takes under a second. You (Claude)
take the role of MiroFish's ReportAgent and write the narrative from its output.

## How MiroFish works (what was borrowed)

MiroFish: seed material → knowledge graph → thousands of agents with
personas → time-stepped social simulation (Twitter/Reddit-like) → ReportAgent
writes the prediction. Each agent has an `AgentActivityConfig`:
`activity_level`, `active_hours`, `sentiment_bias`, `stance`,
`influence_weight`. Each round, `get_active_agents_for_round` activates agents
stochastically, with peak and off-peak multipliers. Posts spread by
recency/popularity weights, an echo-chamber strength and a viral
threshold. "God's-eye" scheduled events can be injected. The expensive
part is that every agent action is an LLM call (plus Zep memory). That
part is replaced here.

## Mapping

| MiroFish | Jyotish Swarm |
|---|---|
| Agent persona | Each natal graha (9) + transit agents (Jupiter, Saturn, Rahu, Ketu) |
| `activity_level` / `active_hours` | Vimshottari activation: MD lord 1.0, AD 0.7, PD 0.35, others 0.1 background |
| `sentiment_bias` | Functional nature for the Lagna (yogakaraka, trikona/kendra/dusthana lordship, kendradhipati) + natural tilt + dignity + vargottama/D9 + combustion |
| `stance` per topic | Relevance to each life area: owns, occupies or aspects its houses (from the Lagna, and at half weight from the Moon) + karaka + varga confirmation (D10 career, D9 marriage, ...) |
| `influence_weight` | Strength: dignity, house type, digbala, cheshta (retrograde), SAV of its sign, own BAV, combustion |
| Social graph & `echo_chamber_strength` | Conjunctions and mutual aspects. Natural friends amplify each other, enemies dampen |
| Peak-hour multiplier | **Double transit** (Jupiter + Saturn both on an area's primary house from the same reference) ×1.3. Noise ×1.5 during dasha sandhi |
| Recommendation recency weight | Exponential momentum of each area's signal (`--recency`) |
| `viral_threshold` | When an area's momentum crosses `--threshold`, an "event" manifests (positive or challenging). 6-month cooldown |
| Scheduled events / god's-eye injection | `--event "YYYY-MM:area:+0.5:note"` (user plans or decisions), `--remedy Planet` (softens that planet's negative output by 30%, a modelling assumption) |
| Many simulations | Monte Carlo `--runs`, sampling birth-time uncertainty (`--time-uncertainty` ± minutes, so different Lagnas are possible) + activation and sentiment noise |
| ReportAgent (LLM) | **You**, writing from the output |

Signals are **relative to the person's own 10-year normal** (a z-score
against a deterministic baseline over ±5 years). So ▲ means "better than
usual for you", not "good in absolute terms". The **natal promise**
line gives the absolute picture from the birth chart. Read both together:
a strong natal promise during a ▼ period means a temporary dip, while a
weak promise during a ▲ period means a relatively better phase in a harder area.

## Running it

```bash
python scripts/swarm_predict.py --date 1990-05-14 --time 10:30 --tz Asia/Kolkata \
  --lat 28.61 --lon 77.21 --months 24 --runs 1000 --time-uncertainty 10 \
  [--areas career,wealth,relationships] [--start 2027-01] \
  [--event "2027-03:career:+0.6:launching a startup"] [--remedy Saturn] [--json]
```

Areas: career, wealth, relationships, health, education, property, children,
travel_foreign, spirituality.

Output: the agent table (sentiment and influence), natal promise per area, a
month × area heatmap, **key windows** (the probability, across all simulated
worlds, that an event manifests in a month, with the top driving agents),
and the timeline context (dasha MD/AD/PD, double transits, Sade Sati,
sandhi). `--json` adds per-month mean, sd, consensus (|mean|/sd, which is
MiroFish's "swarm agreement") and agent stances.

## Writing the report (your ReportAgent role)

1. Run `vedic_chart.py chart` first, so you understand the chart behind the numbers.
2. Run the swarm for the user's horizon and areas. If they mention plans,
   inject them as events, and compare runs with and without an event or
   remedy ("what-if" scenarios, which is MiroFish's core use case).
3. Explain **why** each window appears, using the drivers and timeline
   context ("March–April 2028: Saturn AD inside Rahu MD, plus Saturn's
   transit over your 7th from the Moon, so relationships carry the most
   pressure; 21% of simulated worlds produced a manifest event").
4. Report probabilities as "in X% of simulated scenarios", never as
   certainties. Low consensus (< 0.5) means the swarm disagrees, so say
   the period is open or uncertain.
5. If `lagna_samples` lists more than one sign, the birth time is critical.
   Say so, and offer rectification.
6. Always include the disclaimer: this is a structured simulation of
   classical Jyotish rules, not a scientifically validated forecast. Keep
   the ethics rules from SKILL.md (no death or medical predictions, no
   fear-mongering, end on agency and remedies).

## Tuning knobs (for advanced users)

`--threshold` (event sensitivity, default 0.55), `--echo` (planet-to-planet
influence, 0.5), `--recency` (momentum, 0.4), `--noise` (0.2), `--seed`
(reproducibility). The weights in `AREAS`, `functional_sentiment` and the
strength formula sit at the top of the script, so the model can be
recalibrated against known life events (back-testing: run `--start` in
the past and compare with what actually happened).
