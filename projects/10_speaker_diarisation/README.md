# Speaker Diarisation Deployment

**Domain:** Audio Intelligence Platform  
**Type:** MLOps · Async Inference · Large File Processing

## Problem
Real-time inference deployment of a speaker diarisation model failed due to memory and latency constraints on large audio files.

## Solution Architecture
```
Audio File Upload ──► Object Storage
                              │
                              ▼
                    Async Inference Queue
                              │
                              ▼
                 HuggingFace Diarisation Model
                 (Optimised instance sizing)
                              │
                              ▼
                   Speaker Segments Output ──► Downstream Applications
```

## Key Design Decisions
- **Asynchronous processing** decouples upload from inference — handles large files without timeout
- Instance sizing matched to peak memory requirements of diarisation model
- Output stored in object storage for downstream retrieval

## Techniques
- Async Inference · HuggingFace Model Deployment
- Large File Processing · Speaker Diarisation · MLOps
