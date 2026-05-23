# 🚀 ML Production Portfolio

> **20 years of end-to-end Machine Learning systems** — from research to production — across global e-commerce, government platforms, renewable energy, edtech, and audio intelligence.

---

## 👤 Professional Overview

I specialise in building **scalable ML systems** that take models from research into production and integrate them with operational workflows.

**Three core focus areas:**
- 🏗️ Building end-to-end ML systems — data pipelines to deployed inference endpoints
- 🤖 Developing ML models and integrating them into operational workflows
- ⚡ Scaling AI systems reliably in production with monitoring and feedback loops

---

## 📊 Portfolio at a Glance

| # | Project | Domain | Core Technique |
|---|---------|--------|----------------|
| 1 | [Seller Recommendation & Messaging System](#1-seller-recommendation--messaging-system) | E-Commerce | LLM Reasoning + Behavioural Signals |
| 2 | [Government Procurement AI Assistant](#2-government-procurement-ai-assistant) | Gov Platform | Multi-Agent RAG + Operational Agents |
| 3 | [Program Adoption Propensity Model](#3-program-adoption-propensity-model) | E-Commerce | Gradient Boosting + Precision@K |
| 4 | [Elasticity-Driven Pricing Optimisation](#4-elasticity-driven-pricing-optimisation) | Logistics | LightGBM + Constrained Optimisation |
| 5 | [Solar Plant Anomaly Detection](#5-solar-plant-anomaly-detection) | Renewable Energy | LSTM Encoder-Decoder Autoencoder |
| 6 | [Predictive Maintenance System](#6-predictive-maintenance-system) | Renewable Energy | Supervised Failure Classification |
| 7 | [Remaining Useful Life Estimation](#7-remaining-useful-life-estimation) | Renewable Energy | Regression + Degradation Features |
| 8 | [LLM Inference Optimisation](#8-llm-inference-optimisation) | Digital Publishing | Model Compression 80GB → 18GB |
| 9 | [Question Classification Pipeline](#9-question-classification-pipeline) | EdTech | Embedding + Generative Reasoning |
| 10 | [Speaker Diarisation Deployment](#10-speaker-diarisation-deployment) | Audio AI | Async Inference + HuggingFace |
| 11 | [Speech-to-Text Infrastructure](#11-speech-to-text-infrastructure) | Audio / EdTech | Whisper Async + GPU Optimisation |
| 12 | [RAG Evaluation Framework](#12-rag-evaluation-framework) | AI Platform | LLM-as-Judge + Quality Scoring |
| 13 | [Mathematical Image Generation](#13-mathematical-image-generation-hackathon) | EdTech Hackathon | LLM → Code Gen → Programmatic Render |

---

## 🔍 Detailed Projects

---

### 1. Seller Recommendation & Messaging System
**Domain:** Global E-Commerce Marketplace

#### Challenge
- Needed a scalable system to generate personalised communications for millions of sellers to increase adoption of growth programs (fulfilment, advertising, deals, enhanced content).
- Existing marketing relied on rule-based segmentation and static templates — generic messaging misaligned with each seller's business context.
- Cold-start challenge: no historical training data to determine which program to recommend next in a seller's lifecycle.

#### Solution
- Built a hyper-personalised messaging framework combining **behavioural signal detection** with **LLM reasoning**.
- Constructed seller context using product categories, performance metrics, adoption history, platform activity, and pricing signals.
- LLM reasoning pipeline determined the most relevant program recommendation and generated personalised email content.
- Secondary model generated optimised subject lines to improve deliverability.
- System operated in **sequential engagement cycles**, updating recommendations based on seller responses — adaptive program orchestration across multiple weeks.

**Tech:** LLM Reasoning · Behavioural Signal Processing · Personalisation Engine · Sequential Recommendation

---

### 2. Government Procurement AI Assistant
**Domain:** Government e-Procurement Platform

#### Challenge
- Platform relied on a static FAQ chatbot limited to document-based procurement questions.
- Users needed to perform operational tasks: bid submission status, order retrieval, support tickets, escalations.
- Required transformation from a FAQ bot into an intelligent system combining knowledge retrieval with operational execution.

#### Solution
- Designed an **AI agent architecture** using agent orchestration patterns and tool-based workflows.
- Extended a retrieval-augmented knowledge assistant with **operational agents** interacting with backend services.
- Central orchestration layer handled intent detection and delegated to specialised agents for:
  - Document retrieval (RAG)
  - Database queries (bid/order status)
  - Ticket management and escalation workflows
- Supported **multilingual voice interaction** and fallback routing to human agents.

**Tech:** Multi-Agent Architecture · RAG · Agent Orchestration · Tool-Use · Intent Detection · Voice AI

---

### 3. Program Adoption Propensity Model
**Domain:** Global E-Commerce Marketplace

#### Challenge
- Seller growth team needed a data-driven method to identify sellers most likely to adopt a warehousing and distribution program across global marketplaces.
- Existing heuristic targeting resulted in inefficient campaign spend and low adoption rates.

#### Solution
- Developed an ML **propensity model** using behavioural, transactional, and operational seller data.
- Feature engineering: fulfilment patterns, seller tenure, shipment volumes, listing activity, engagement signals.
- Addressed class imbalance using sampling techniques and robust evaluation metrics.
- Evaluated with **AUC-ROC** and **Precision@Top-K** for high-precision targeting within budget constraints.
- Validated via **controlled A/B experiments** — demonstrated improved campaign efficiency and higher program adoption.

**Tech:** Gradient Boosting · Feature Engineering · Class Imbalance Handling · AUC-ROC · Precision@K · A/B Testing

---

### 4. Elasticity-Driven Pricing Optimisation
**Domain:** Cross-Border Logistics Platform

#### Challenge
- Pricing relied on flat markup rules ignoring demand elasticity across shipping routes and seller cohorts.
- Caused pricing inefficiencies reducing revenue or discouraging adoption.

#### Solution
- Built an ML **pricing engine** modelling demand response to price changes using historical shipment data.
- **LightGBM** regression predicted shipment volume as a function of price and operational signals.
- Applied **monotonic constraints** to maintain economically consistent demand behaviour.
- Performed **constrained revenue optimisation** across markup scenarios with adoption guardrails.
- Weekly pricing tables deployed to production via a low-latency lookup mechanism.

**Tech:** LightGBM · Demand Elasticity Modelling · Monotonic Constraints · Revenue Optimisation · Real-Time Lookup

---

### 5. Solar Plant Anomaly Detection
**Domain:** Renewable Energy Company

#### Challenge
- Solar plants generate large volumes of telemetry from environmental sensors and equipment monitoring.
- Rule-based monitoring produced excessive false alarms and missed complex cross-signal relationships.

#### Solution
- Developed a **multivariate anomaly detection system** using an LSTM encoder-decoder autoencoder.
- Trained on environmental and operational signals: irradiance, temperature, wind speed, generation output.
- Model learned normal temporal relationships; anomalies detected via **reconstruction error thresholds**.
- Data pipeline included signal cleaning, feature engineering, and sliding time windows.
- Delivered early alerts for abnormal plant behaviour, improving operational monitoring reliability.

**Tech:** LSTM Encoder-Decoder · Autoencoder · Multivariate Time-Series · Reconstruction Error · Anomaly Detection

---

### 6. Predictive Maintenance System
**Domain:** Renewable Energy Company

#### Challenge
- Inverters and transformers occasionally failed without warning, causing unplanned downtime and costly interventions.

#### Solution
- Built a **predictive maintenance model** forecasting equipment failures 24 hours to 7 days in advance.
- Formulated as **supervised classification** — predicting failure probability within a future time window.
- Engineered labels representing pre-failure conditions from historical telemetry.
- Features: temperature signals, inverter power metrics, generation output, operational fault indicators.
- Model produced **asset-level risk scores** enabling proactive maintenance scheduling.

**Tech:** Supervised Classification · Failure Label Engineering · Telemetry Feature Engineering · Risk Scoring

---

### 7. Remaining Useful Life Estimation
**Domain:** Renewable Energy Company

#### Challenge
- Operations teams needed estimates of how long equipment could operate before reaching failure conditions.

#### Solution
- Developed a **regression-based RUL prediction system** using solar plant equipment telemetry.
- Engineered RUL target by resetting counter at each failure event and counting backward.
- Features: transformer temperatures, oil temperature, inverter output power, plant generation signals.
- System predicted **time-to-failure** enabling maintenance planning based on degradation trends.

**Tech:** Regression · RUL Feature Engineering · Degradation Modelling · Predictive Maintenance

---

### 8. LLM Inference Optimisation
**Domain:** Digital Publishing Platform

#### Challenge
- Generative AI application required large GPU instances and suffered high inference latency due to model size (~80 GB).

#### Solution
- Optimised model and deployment pipeline using **model compression** and inference optimisation techniques.
- Reduced model size from **~80 GB → ~18 GB** by restructuring training artefacts and optimising configuration.
- Optimised model ran on **smaller GPU instances** without compromising generation quality.
- Implemented custom container images and **auto-scaling inference endpoints** — improved latency and reduced infrastructure costs.

**Tech:** Model Compression · Quantisation · Inference Optimisation · Auto-Scaling Endpoints · Container Optimisation

---

### 9. Question Classification Pipeline
**Domain:** EdTech Platform

#### Challenge
- Required scalable system to categorise educational questions into subject areas for efficient routing within the learning platform.

#### Solution
- Designed a **classification pipeline** combining embedding-based retrieval with generative model reasoning.
- Questions converted to embeddings and matched against candidate categories.
- **Generative model** refined classification decisions when ambiguity existed.
- Deployed using cloud ML platform endpoints with optimised containers and production monitoring.

**Tech:** Embedding Retrieval · Generative Reasoning · Classification Pipeline · SageMaker-style Endpoints

---

### 10. Speaker Diarisation Deployment
**Domain:** Audio Intelligence Platform

#### Challenge
- Needed to deploy a speaker diarisation model for multi-speaker audio recordings.
- Real-time inference deployment failed due to memory and latency constraints on large files.

#### Solution
- Redesigned deployment using **asynchronous inference endpoints**.
- HuggingFace diarisation model packaged with appropriate instance sizing for large audio files.
- Architecture enabled reliable **speaker segmentation** for long recordings with improved stability.
- Documented deployment approach as a technical reference for other teams.

**Tech:** Async Inference · HuggingFace · Speaker Diarisation · SageMaker-style Async · Large File Processing

---

### 11. Speech-to-Text Infrastructure
**Domain:** Large-Scale Transcription Platform

#### Challenge
- Organisations needed scalable infrastructure for Whisper-based speech-to-text at large transcription volumes.
- Real-time hosting impractical due to model size and inference time for audio.

#### Solution
- Designed a scalable **Whisper-based speech-to-text architecture** using asynchronous endpoints.
- Audio uploaded to storage; processed asynchronously by the Whisper model.
- Supported large audio files and **batch transcription pipelines** with efficient GPU utilisation.
- Created deployment scripts and reference architectures for team replication.

**Tech:** Whisper · Async Inference · GPU Optimisation · Batch Transcription · Scalable Architecture

---

### 12. RAG Evaluation Framework
**Domain:** AI Platform

#### Challenge
- Evaluating RAG system response quality at production scale was a major bottleneck.
- Manual evaluation couldn't scale with large volumes of generated responses.

#### Solution
- Designed an **automated LLM-as-Judge evaluation framework** for RAG outputs.
- Quality dimensions: citation relevance, keyword alignment, factual accuracy, completeness, correctness, hallucination detection.
- Framework extracted claims from generated responses, verified against retrieved documents, and aggregated into a **unified quality score**.
- Enabled continuous monitoring and systematic improvement of retrieval and generation quality.

**Tech:** LLM-as-Judge · RAG Evaluation · Hallucination Detection · Claim Verification · Quality Scoring

---

### 13. Mathematical Image Generation (Hackathon)
**Domain:** EdTech — Global Hackathon

#### Challenge
- Teams challenged to generate mathematically accurate diagrams from textual math questions for an educational platform.
- Diffusion-based image models failed to produce geometrically correct shapes, aligned edges, and labelled components.

#### Solution
- Approached as a **reasoning and rendering problem**, not a pure image generation problem.
- Designed a **three-stage pipeline:**
  1. **LLM Reasoning** — analysed math question, identified geometric concepts
  2. **Code Generation** — LLM generated Python code constructing the diagram (points, edges, annotations)
  3. **Programmatic Rendering** — code executed using Matplotlib/Seaborn to produce precise, reproducible diagrams
- Implemented **automated validation** — LLM reviewed generated code before execution.
- Architecture demonstrated that LLM reasoning + programmatic graphics outperforms generative image models for structured domains.

**Tech:** LLM Reasoning · Code Generation · Matplotlib · Seaborn · Programmatic Rendering · Automated Validation

---

## 🛠️ Technical Skills Summary

| Category | Technologies |
|----------|-------------|
| **LLM & Generative AI** | LLM Reasoning, RAG, Prompt Engineering, LLM-as-Judge, Inference Optimisation |
| **Agentic AI** | Multi-Agent Architecture, Agent Orchestration, Tool-Use, Intent Detection |
| **Classical ML** | Gradient Boosting, LightGBM, Regression, Classification, Propensity Modelling |
| **Deep Learning** | LSTM, Encoder-Decoder, Autoencoders, Transformers, Embeddings |
| **Time Series** | Anomaly Detection, Predictive Maintenance, RUL Estimation, Forecasting |
| **MLOps & Deployment** | Async Inference, Auto-Scaling, Container Optimisation, Model Compression |
| **Evaluation** | A/B Testing, AUC-ROC, Precision@K, RAGAS, Hallucination Detection |

---

## 📁 Repository Structure

```
ml-production-portfolio/
├── README.md                          # This file — full portfolio overview
├── projects/
│   ├── 01_seller_recommendation/      # LLM + Behavioural Signals
│   ├── 02_procurement_ai_assistant/   # Multi-Agent RAG
│   ├── 03_propensity_model/           # Gradient Boosting
│   ├── 04_pricing_optimisation/       # LightGBM + Elasticity
│   ├── 05_anomaly_detection/          # LSTM Autoencoder
│   ├── 06_predictive_maintenance/     # Failure Classification
│   ├── 07_rul_estimation/             # RUL Regression
│   ├── 08_llm_inference_optimisation/ # Model Compression
│   ├── 09_question_classification/    # Embedding + GenAI
│   ├── 10_speaker_diarisation/        # Async Inference
│   ├── 11_speech_to_text/             # Whisper Infrastructure
│   ├── 12_rag_evaluation/             # LLM-as-Judge
│   └── 13_math_image_generation/      # Code Gen + Rendering
└── architecture/
    └── diagrams/                      # System architecture diagrams
```

---

*Built with 20 years of production ML experience across global enterprise systems.*
