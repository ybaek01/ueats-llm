"""
runs/ gather issues.json files
Compose consolidated_report.pdf
"""
import argparse, json, os, pathlib, subprocess, tempfile, re
from jinja2 import Template

TEMPLATE_HTML = """
<!doctype html><html><head>
<meta charset="utf-8"><style>
body{font-family:Arial,Helvetica,sans-serif;margin:40px;}
h1{page-break-before:always;}
img{max-width:350px;border:1px solid #ccc;}
pre{white-space:pre-wrap;background:#f5f5f5;padding:10px;border-radius:4px;}
</style></head><body>
<h1>UEATS-LLM Consolidated Report</h1>
<p>Total personas: {{items|length}}</p>

{% for it in items %}
<h2>{{loop.index}}. {{it.persona.name}}</h2>
<p><strong>Condition:</strong> {{it.persona.condition}}</p>
<img src="{{it.png_path}}" alt="screenshot">
<pre>{{it.analysis}}</pre>
{% endfor %}
</body></html>
"""

def gather_runs(root_dir: pathlib.Path):
    """runs/ 하위의 issues.json → list(dict) 로 수집 + 정렬"""
    items = []
    for json_path in root_dir.rglob("issues.json"):
        data = json.load(open(json_path))
        items.append({
            "persona": data["persona"],
            "analysis": data["analysis"],
            "png_path": os.path.relpath(json_path.with_name("final.png"), root_dir)
        })

    # ── 정렬: condition 순서 → ID 숫자 순 ──
    order = {"uniform": 0, "diet": 1, "diverse": 2}
    def sort_key(it):
        cond = it["persona"].get("condition", "")
        raw  = it["persona"].get("id", "")        # 예: U-07, D-12, F-03
        m = re.search(r"(\d+)", raw)
        num = int(m.group(1)) if m else 0
        return (order.get(cond, 99), num)

    return sorted(items, key=sort_key)

def main(args):
    root = pathlib.Path(args.input)
    items = gather_runs(root)

    html = Template(TEMPLATE_HTML).render(items=items)
    tmp_html = tempfile.NamedTemporaryFile(delete=False, suffix=".html").name
    with open(tmp_html, "w") as f:
        f.write(html)

    subprocess.run(["wkhtmltopdf", tmp_html, args.pdf], check=True)  # wkhtmltopdf 필요
    os.remove(tmp_html)
    print(f"✅ PDF Saved: {args.pdf}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="runs", help="runs 폴더")
    ap.add_argument("--pdf",   default="consolidated_report.pdf")
    main(ap.parse_args())
