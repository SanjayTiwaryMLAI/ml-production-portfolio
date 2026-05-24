"""
Use Case: Mathematical Image Generation — LLM → Code → Render
--------------------------------------------------------------
Problem : Diffusion models fail to produce accurate geometric diagrams.
Approach: LLM concept extraction → Python code generation →
          Automated validation → Matplotlib rendering.
"""
import re, json
from pathlib import Path
from openai import OpenAI

client = OpenAI()


def extract_concepts(question):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":
            f"Identify geometric/mathematical concepts to visualize.

Q: "{question}"

"
            f"JSON: {{"type":"triangle|circle|coordinate|function|vector|other","
            f""elements":["..."],"labels":["..."],"description":"..."}}"}],
        temperature=0, response_format={"type":"json_object"})
    return json.loads(resp.choices[0].message.content)


def generate_code(question, concepts, output_path):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":
            f"Write Python code to create an accurate mathematical diagram using matplotlib.

"
            f"Question: "{question}"
Concepts: {json.dumps(concepts)}

"
            f"Requirements:
"
            f"1. Import matplotlib.pyplot as plt and numpy as np
"
            f"2. Create a clear, labeled, mathematically accurate diagram
"
            f"3. Save to '{output_path}' with dpi=150, then plt.close()
"
            f"Write ONLY Python code, no explanations."}],
        temperature=0.2)
    code = resp.choices[0].message.content
    return re.sub(r"^```python
|^```
|```$", "", code, flags=re.MULTILINE).strip()


def validate_code(code):
    issues = []
    if "matplotlib" not in code: issues.append("Missing matplotlib import")
    if "savefig" not in code:    issues.append("Missing plt.savefig()")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":
            f"Review this diagram code. Will it run? Any bugs?

{code}

"
            f"JSON: {{"valid":true/false,"issues":[...]}}"}],
        temperature=0, response_format={"type":"json_object"})
    data = json.loads(resp.choices[0].message.content)
    issues.extend(data.get("issues", []))
    return {"valid": len(issues)==0 and data.get("valid",False), "issues": issues}


def render(code, output_path):
    try:
        exec(compile(code, "<generated>", "exec"), {})
        return {"success": True, "path": output_path}
    except Exception as e:
        return {"success": False, "error": str(e)}


def generate_diagram(question, output_dir="./diagrams", max_retries=2):
    Path(output_dir).mkdir(exist_ok=True)
    output_path = f"{output_dir}/diagram_{abs(hash(question))}.png"

    print(f"  Extracting concepts...")
    concepts = extract_concepts(question)
    print(f"  Type: {concepts.get('type')} | Elements: {concepts.get('elements', [])}")

    for attempt in range(1, max_retries+1):
        print(f"  Generating code (attempt {attempt})...")
        code = generate_code(question, concepts, output_path)

        val = validate_code(code)
        if not val["valid"]:
            print(f"  Issues: {val['issues']}")
            if attempt == max_retries:
                return {"success": False, "issues": val["issues"]}
            continue

        result = render(code, output_path)
        if result["success"]:
            return {"success": True, "path": result["path"], "concepts": concepts}
        print(f"  Render error: {result['error']}")

    return {"success": False, "error": "Failed after retries"}


if __name__ == "__main__":
    questions = [
        "Draw a right triangle ABC with AB=3, BC=4, hypotenuse AC=5, label all sides",
        "Plot sin(x) and cos(x) from 0 to 2π on the same graph with legend",
        "Draw a unit circle with angles 30, 45, 60, 90 degrees marked",
    ]
    print("Mathematical Image Generation Pipeline")
    print("="*50)
    for q in questions:
        print(f"\nQ: {q}")
        r = generate_diagram(q)
        if r["success"]:
            print(f"  ✅ Saved: {r['path']}")
        else:
            print(f"  ❌ Failed: {r.get('error', r.get('issues'))}")
