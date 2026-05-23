# Predictive Maintenance System

**Domain:** Renewable Energy Company  
**Type:** Supervised Classification · Failure Prediction

## Problem
Unplanned inverter and transformer failures caused costly downtime. Needed 24-hour to 7-day advance warning to enable proactive maintenance.

## Label Engineering
```
Raw Telemetry ──► Failure Event Detection ──► Pre-Failure Window Labelling
                                               (e.g., 24h, 48h, 7-day windows)
                                                        │
                                            Binary Label: WILL_FAIL / NORMAL
```

## Features
- Temperature signals (ambient, component-level)
- Inverter power metrics (AC/DC output, conversion efficiency)
- Generation output (actual vs. expected)
- Operational fault indicators and error codes

## Output
**Asset-level risk scores** → Maintenance team scheduling dashboard

## Key Design Decisions
- Multiple horizon labels (24h, 7-day) for flexible maintenance planning
- Pre-failure window size tuned to balance precision and lead time
- Asset-level scoring enables prioritisation across a fleet of equipment

## Techniques
- Supervised Classification · Failure Label Engineering
- Telemetry Feature Engineering · Risk Scoring · Fleet-Level Monitoring
