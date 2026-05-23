# Solar Plant Anomaly Detection

**Domain:** Renewable Energy Company  
**Type:** Unsupervised Anomaly Detection · Multivariate Time-Series

## Problem
Rule-based monitoring generated excessive false alarms and missed complex cross-signal anomalies in solar plant telemetry data.

## Model Architecture
```
Input Signals: [Irradiance, Temperature, Wind Speed, Generation Output, ...]
        │
        ▼
Sliding Window (time-series segmentation)
        │
        ▼
LSTM Encoder ──► Compressed Latent Representation ──► LSTM Decoder
        │
        ▼
Reconstruction Error Computation
        │
        ▼
Threshold-Based Anomaly Flagging ──► Early Alert System
```

## Key Design Decisions
- **Encoder-decoder** learns normal temporal relationships without labelled anomalies
- **Reconstruction error** as anomaly score — no need for labelled failure data
- Sliding window captures multi-step temporal dependencies
- Signal cleaning + feature engineering critical for training stability

## Techniques
- LSTM Encoder-Decoder · Autoencoder · Reconstruction Error
- Multivariate Time-Series · Unsupervised Anomaly Detection
