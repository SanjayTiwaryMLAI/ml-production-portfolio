# LLM Inference Optimisation

**Domain:** Digital Publishing Platform  
**Type:** Model Compression · Inference Engineering

## Problem
~80 GB generative AI model required large GPU instances and produced high inference latency — making production deployment cost-prohibitive.

## Optimisation Results
| Metric | Before | After |
|--------|--------|-------|
| Model Size | ~80 GB | ~18 GB |
| GPU Instance | Large (expensive) | Smaller (cost-effective) |
| Inference Latency | High | Reduced |
| Generation Quality | Baseline | Maintained |

## Approach
- Restructured training artefacts and model configuration
- Model compression without fine-tuning quality degradation
- Custom container images for optimised runtime
- Auto-scaling inference endpoints for variable load

## Key Design Decisions
- Compression focused on artefact restructuring rather than aggressive quantisation to preserve quality
- Auto-scaling handles traffic spikes without over-provisioning

## Techniques
- Model Compression · Inference Optimisation · Container Engineering
- Auto-Scaling Endpoints · GPU Cost Optimisation
