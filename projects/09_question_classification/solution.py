"""
Use Case: Educational Question Classification Pipeline
------------------------------------------------------
Problem : Categorise educational questions into subject areas at scale.
Approach: Embedding retrieval (Top-K) → Generative disambiguation.
"""
import json
import numpy as np
from openai import OpenAI

client = OpenAI()

CATEGORIES = {
    "Mathematics":      ["algebra","geometry","calculus","trigonometry","statistics","integration","derivative","equation","matrix","probability"],
    "Physics":          ["force","motion","energy","gravity","optics","electricity","magnetism","thermodynamics","momentum","quantum"],
    "Chemistry":        ["reaction","element","compound","molecule","bond","acid","base","periodic","oxidation","electrolysis"],
    "Biology":          ["cell","dna","evolution","genetics","photosynthesis","respiration","ecosystem","enzyme","osmosis","taxonomy"],
    "Computer Science": ["algorithm","recursion","sorting","complexity","binary","graph","tree","stack","queue","data structure"],
    "History":          ["war","civilization","empire","revolution","dynasty","colonialism","independence","treaty","monarchy","renaissance"],
    "Economics":        ["supply","demand","inflation","gdp","market","trade","fiscal","monetary","elasticity","opportunity cost"],
}


class EmbeddingRetriever:
    def __init__(self):
        self.index = {}
        for cat, kws in CATEGORIES.items():
            text = f"{cat}: {', '.join(kws)}"
            resp = client.embeddings.create(input=text, model="text-embedding-3-small")
            self.index[cat] = np.array(resp.data[0].embedding)
        print(f"Index built: {len(self.index)} categories")

    def retrieve(self, question, top_k=3):
        resp = client.embeddings.create(input=question, model="text-embedding-3-small")
        q    = np.array(resp.data[0].embedding)
        sims = {cat: float(np.dot(q, emb) / (np.linalg.norm(q)*np.linalg.norm(emb)+1e-8))
                for cat, emb in self.index.items()}
        return sorted(sims.items(), key=lambda x: -x[1])[:top_k]


class Pipeline:
    def __init__(self):
        self.retriever = EmbeddingRetriever()

    def classify(self, question):
        candidates = self.retriever.retrieve(question, top_k=3)
        top, second = candidates[0][1], candidates[1][1] if len(candidates)>1 else 0

        # Fast path: clear winner
        if top - second > 0.15:
            return {"category": candidates[0][0], "confidence": round(top,4), "method": "retrieval"}

        # Disambiguation via LLM
        prompt = (f"Classify this educational question into one subject.

"
                  f"Question: "{question}"

"
                  f"Candidates:
" + "
".join(f"- {c} ({s:.3f})" for c,s in candidates) +
                  f"

Choose from: {', '.join(CATEGORIES.keys())}
"
                  f"JSON: {{"category": "<category>", "reason": "<reason>"}}")
        resp = client.chat.completions.create(model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}], temperature=0,
            response_format={"type":"json_object"})
        data = json.loads(resp.choices[0].message.content)
        return {"category": data["category"], "confidence": round(top,4),
                "reason": data.get("reason",""), "method": "generative"}


if __name__ == "__main__":
    pipe = Pipeline()
    tests = [
        "What is the derivative of sin(x)?",
        "Explain Newton's third law of motion.",
        "What is the difference between mitosis and meiosis?",
        "How does binary search work?",
        "What caused the French Revolution?",
        "Explain price elasticity of demand.",
    ]
    print("\nQuestion Classification Results:")
    for q in tests:
        r = pipe.classify(q)
        print(f"  Q: {q}")
        print(f"     → {r['category']} (conf: {r['confidence']}, method: {r['method']})")
