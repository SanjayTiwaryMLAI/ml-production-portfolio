# Mathematical Image Generation

**Domain:** EdTech — Global Hackathon  
**Type:** LLM Reasoning · Code Generation · Programmatic Rendering

## Problem
Generate mathematically accurate diagrams from textual math questions. Diffusion-based image generation models failed to produce geometrically correct diagrams with proper labels and aligned edges.

## Key Insight
> Treat as a **reasoning and rendering problem**, not an image generation problem.

## Three-Stage Pipeline
```
Math Question Text
        │
        ▼
┌─────────────────────────────┐
│  Stage 1: LLM Reasoning     │
│  - Identify geometric       │
│    concepts (triangle,      │
│    vectors, angles, etc.)   │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Stage 2: Code Generation   │
│  - LLM generates Python     │
│    code for diagram         │
│  - Defines points, edges,   │
│    annotations              │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Stage 3: Validation &      │
│           Rendering         │
│  - LLM validates code       │
│    (imports, executability) │
│  - Execute → Matplotlib /   │
│    Seaborn renders diagram  │
└─────────────┬───────────────┘
              │
              ▼
   Precise Mathematical Diagram
   ──► Cloud Storage ──► Question Linking
```

## Why This Approach Won
- **Deterministic rendering** — same input always produces same output
- **Mathematically precise** — programmatic geometry vs. probabilistic pixels
- **Reproducible** — diagrams can be regenerated and version-controlled
- **Scalable** — code generation scales with LLM inference

## Techniques
- LLM Reasoning · Code Generation · Programmatic Rendering
- Matplotlib · Seaborn · Automated Code Validation
