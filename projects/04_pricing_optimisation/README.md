# Elasticity-Driven Pricing Optimisation

**Domain:** Cross-Border Logistics Platform  
**Type:** Demand Modelling · Revenue Optimisation

## Problem
Replace flat markup pricing rules with an elasticity-aware system that optimises revenue while maintaining adoption rates across seller cohorts and shipping routes.

## System Design
```
Historical Shipment Data
        │
        ▼
Seller Cohort Segmentation
        │
        ▼
LightGBM Demand Model
(Price → Volume, with monotonic constraints)
        │
        ▼
Constrained Revenue Optimisation
(Maximise revenue subject to adoption guardrails)
        │
        ▼
Weekly Pricing Tables ──► Low-Latency Production Lookup
```

## Key Design Decisions
- **Monotonic constraints** ensure economically consistent demand curves (higher price = lower/equal volume)
- Cohort-level modelling captures heterogeneous price sensitivity
- Constrained optimisation prevents adoption collapse at high markups
- Weekly batch generation with real-time lookup for booking queries

## Techniques
- LightGBM · Demand Elasticity · Monotonic Constraints
- Constrained Optimisation · Revenue Maximisation · Real-Time Lookup
