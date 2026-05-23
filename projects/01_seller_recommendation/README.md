# Seller Recommendation & Messaging System

**Domain:** Global E-Commerce Marketplace  
**Type:** Personalisation · LLM Reasoning · Sequential Recommendation

## Problem
Generate hyper-personalised seller communications at scale to drive adoption of growth programs — with no historical training data (cold-start).

## Architecture
```
Seller Signals ──► Context Builder ──► LLM Reasoning ──► Program Recommender
                                                     └──► Email Generator
                                                     └──► Subject Line Optimiser
                                                            │
                                              Engagement Feedback Loop
```

## Key Design Decisions
- Behavioural signals (platform activity, pricing actions) used as real-time context
- LLM reasoning over seller profile rather than static rule matching
- Sequential engagement cycles enable adaptive recommendation over weeks
- Secondary model for subject line optimisation improves deliverability

## Techniques
- LLM Reasoning Pipeline
- Behavioural Signal Extraction
- Personalised Content Generation
- Sequential / Lifecycle-based Recommendation
- Engagement Feedback Loop
