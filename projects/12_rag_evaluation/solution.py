"""
Use Case: Automated RAG Evaluation Framework — LLM-as-Judge
------------------------------------------------------------
Problem : Manual evaluation of RAG outputs cannot scale in production.
Approach: Claim extraction → Hallucination detection → Multi-dimension
          quality scoring → Unified quality score.
"""
import json
from dataclasses import dataclass, field
from openai import OpenAI

client = OpenAI()


@dataclass
class EvalResult:
    question:           str
    answer:             str
    citation_relevance: float = 0.0
    factual_accuracy:   float = 0.0
    completeness:       float = 0.0
    hallucination:      float = 0.0
    overall:            float = 0.0
    flagged:            list  = field(default_factory=list)


def llm_score(prompt):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0,
    )
    try:    return min(1.0, max(0.0, float(resp.choices[0].message.content.strip())))
    except: return 0.5


def extract_claims(answer):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":f"Extract factual claims as JSON list.

Answer: {answer}

JSON: {{"claims":[...]}}"}],
        temperature=0, response_format={"type":"json_object"})
    return json.loads(resp.choices[0].message.content).get("claims", [])


def detect_hallucinations(claims, context):
    flagged = []
    ctx_str = "
".join(context)
    for claim in claims:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":f"Context:
{ctx_str}

Claim: {claim}

SUPPORTED or UNSUPPORTED?"}],
            temperature=0)
        if "UNSUPPORTED" in resp.choices[0].message.content.upper():
            flagged.append(claim)
    return len(flagged)/max(len(claims),1), flagged


def evaluate(question, answer, citations):
    r = EvalResult(question=question, answer=answer)
    claims = extract_claims(answer)

    r.citation_relevance = llm_score(
        f"Rate 0-1 how relevant these docs are to the question.
Q: {question}
Docs:
" + "
".join(f"- {c}" for c in citations) + "
Number only.")
    r.factual_accuracy   = llm_score(
        f"Context:
" + "
".join(citations) + f"

Answer: {answer}

Rate factual accuracy 0-1. Number only.")
    r.completeness       = llm_score(
        f"Rate 0-1 how completely the answer addresses the question.
Q: {question}
A: {answer}
Number only.")
    r.hallucination, r.flagged = detect_hallucinations(claims, citations)

    # Keyword alignment
    qw = set(question.lower().split())
    aw = set(answer.lower().split())
    kw = round(len(qw & aw) / max(len(qw),1), 2)

    r.overall = round(
        0.20 * r.citation_relevance +
        0.25 * r.factual_accuracy +
        0.20 * r.completeness +
        0.25 * (1 - r.hallucination) +
        0.10 * kw, 3)
    return r


if __name__ == "__main__":
    tests = [
        {"question": "What is the capital of France?",
         "answer":   "The capital of France is Paris, known for the Eiffel Tower and world-class cuisine.",
         "citations": ["Paris is the capital of France.", "The Eiffel Tower is located in Paris."]},
        {"question": "What is the boiling point of water?",
         "answer":   "Water boils at 100°C. It also boils at 150°C on Mars.",
         "citations": ["Water boils at 100°C at standard atmospheric pressure."]},
    ]
    print("RAG Evaluation Results:")
    for t in tests:
        r = evaluate(t["question"], t["answer"], t["citations"])
        print(f"\nQ: {r.question}")
        print(f"  Overall:      {r.overall:.3f}")
        print(f"  Factual:      {r.factual_accuracy:.2f} | Hallucination: {r.hallucination:.2f}")
        print(f"  Completeness: {r.completeness:.2f} | Citation:      {r.citation_relevance:.2f}")
        if r.flagged: print(f"  ⚠ Flagged: {r.flagged}")
