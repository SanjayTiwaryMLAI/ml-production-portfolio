# Government Procurement AI Assistant

**Domain:** Government e-Procurement Platform  
**Type:** Multi-Agent AI · RAG · Agentic Workflows

## Problem
Transform a static FAQ chatbot into an intelligent assistant capable of both answering policy questions AND executing operational tasks (bid status, order retrieval, ticket management).

## Architecture
```
User Query
    │
    ▼
Intent Detection (Orchestrator)
    ├──► Knowledge Agent (RAG) ──► Policy/Document Q&A
    ├──► Database Agent ──────────► Bid Status / Order Details
    ├──► Ticket Agent ────────────► Raise / Track Support Tickets
    └──► Escalation Agent ────────► Human Handoff
         │
    Multilingual Voice Interface
```

## Key Design Decisions
- Central orchestration layer for intent routing
- Specialised agents with single responsibility per domain
- RAG for policy knowledge + API calls for operational data
- Multilingual + voice support for accessibility
- Fallback to human agents when confidence is low

## Techniques
- Multi-Agent Orchestration
- Retrieval-Augmented Generation (RAG)
- Tool-Use / Function Calling
- Intent Classification
- Voice AI Integration
