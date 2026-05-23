# Remaining Useful Life (RUL) Estimation

**Domain:** Renewable Energy Company  
**Type:** Regression · Degradation Modelling

## Problem
Operations teams needed time-to-failure estimates for equipment to enable long-horizon maintenance planning beyond binary failure prediction.

## RUL Target Engineering
```
Equipment History:
  t=0 ──── normal ──── normal ──── [FAILURE] ──── normal ──── [FAILURE]
                                       │                          │
RUL Labels:     N ... 3, 2, 1, 0      reset      N ... 3, 2, 1, 0
```

## Features
| Signal | Source |
|--------|--------|
| Transformer temperature | Component sensors |
| Oil temperature | Cooling system sensors |
| Inverter output power | Power monitoring |
| Plant generation output | SCADA system |

## Output
**Days-to-failure prediction** per asset → Maintenance scheduling

## Techniques
- Regression · RUL Label Engineering · Degradation Feature Engineering
- Time-Series Regression · Fleet-Level Prediction
