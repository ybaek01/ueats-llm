"""
Scan runs/*/issues.json and compose consolidated_report.pdf.
Requires: jinja2, wkhtmltopdf
"""
import argparse, json, pathlib, subprocess, tempfile, re
from jinja2 import Template

TEMPLATE_HTML = """
<!doctype html><html><head>
<meta charset="utf-8"><style>
body{font-family:Arial,Helvetica,sans-serif;margin:40px;}
h1{page-break-before:always;}
pre{white-space:pre-wrap;background:#f5f5f5;padding:10px;border-radius:4px;}
.scorebox{
border:2px solid #000;
padding:10px 12px;
border-radius:8px;
display:inline-block;
font-weight:bold;
margin:8px 0;
white-space: nowrap;}
</style></head><body>
<h1>UEATS-LLM Consolidated Report</h1>
<p>Total personas: {{items|length}}</p>

{% for it in items %}
<h2>{{loop.index}}. {{ it.persona.id or it.persona.name }}</h2>
<p><strong>Condition:</strong> {{it.persona.condition}}</p>
{% if it.score is not none %}
<div class="scorebox">{{ '%.1f'|format(it.score) }}&nbsp; / &nbsp;5.0</div>
{% endif %}
{% if it.description %}
<p><strong>Description:</strong> {{ it.description }}</p>
{% endif %}
<pre>{{it.analysis}}</pre>
{% endfor %}
</body></html>
"""

def gather_runs(root_dir: pathlib.Path):
    items = []
    for json_path in root_dir.rglob("issues.json"):
        data = json.load(open(json_path))
        try:
            score = float(data.get("score"))
        except Exception:
            score = None
        items.append({
            "persona": data["persona"],
            "analysis": data["analysis"],
            "description": data.get("description") or "",
            "score": score,
        })

    order = {"uniform": 0, "diet": 1, "diverse": 2}
    def sort_key(it):
        cond = it["persona"].get("condition", "")
        raw  = it["persona"].get("id", "")
        m = re.search(r"(\d+)", raw)
        num = int(m.group(1)) if m else 0
        return (order.get(cond, 99), num)

    return sorted(items, key=sort_key)

def main(args):
    root = pathlib.Path(args.input)
    items = gather_runs(root)

    html = Template(TEMPLATE_HTML).render(items=items)
    tmp_html = tempfile.NamedTemporaryFile(delete=False, suffix=".html").name
    with open(tmp_html, "w", encoding="utf-8") as f:
        f.write(html)

    subprocess.run(["wkhtmltopdf", tmp_html, args.pdf], check=True)
    pathlib.Path(tmp_html).unlink(missing_ok=True)
    print(f"✅ PDF Saved: {args.pdf}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="runs")
    ap.add_argument("--pdf",   default="consolidated_report.pdf")
    main(ap.parse_args())
