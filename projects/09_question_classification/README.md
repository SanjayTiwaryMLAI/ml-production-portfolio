# Question Classification Pipeline

**Domain:** EdTech Platform  
**Type:** Classification · Embedding + Generative Reasoning

## Problem
Categorise educational questions into subject areas at scale for efficient routing within a learning platform.

## Pipeline Architecture
```
Incoming Question
        │
        ▼
Embedding Model ──► Vector Representation
        │
        ▼
Candidate Category Retrieval (Top-K similarity)
        │
        ▼
Generative Model (Disambiguation)
  ├── High confidence ──► Direct assignment
  └── Ambiguous ──────► LLM reasoning over candidates
        │
        ▼
Subject Category Label ──► Routing System
```

## Key Design Decisions
- Two-stage pipeline: fast embedding retrieval + accurate generative disambiguation
- Generative model only invoked for ambiguous cases (cost efficiency)
- Optimised containers for low-latency classification

## Techniques
- Sentence Embeddings · Semantic Similarity · Generative Classification
- Two-Stage Pipeline · Cloud ML Deployment
