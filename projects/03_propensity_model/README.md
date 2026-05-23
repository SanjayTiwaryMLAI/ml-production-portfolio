# Program Adoption Propensity Model

**Domain:** Global E-Commerce Marketplace  
**Type:** Binary Classification · Marketing ML

## Problem
Identify sellers most likely to adopt a warehousing and distribution program to replace inefficient heuristic-based campaign targeting.

## Feature Engineering
| Feature Category | Examples |
|-----------------|---------|
| Fulfilment Patterns | Self-ship rate, returns rate, fulfilment method history |
| Seller Tenure | Account age, program adoption history |
| Shipment Volumes | Monthly shipment count, order volume trends |
| Listing Activity | Active ASINs, listing update frequency |
| Engagement Signals | Seller Central logins, tool usage |

## Evaluation Strategy
- **AUC-ROC** — overall discriminative power
- **Precision@Top-K** — high-precision targeting within budget
- **A/B Testing** — production validation of campaign lift

## Key Design Decisions
- Precision@K chosen over accuracy due to fixed marketing budget constraint
- Sampling techniques to handle class imbalance
- Campaign pipeline integration with controlled experiment validation

## Techniques
- Gradient Boosting · Feature Engineering
- Class Imbalance Handling · AUC-ROC · Precision@K · A/B Testing
