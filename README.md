# Persona Diversity Simulation
Persona Diversity: Comparing Usability-Testing Outcomes across Heterogeneous Personas

A lightweight pipeline that combines Playwright, OpenAI GPT-4o-mini, and JSON persona files to simulate end-to-end food-ordering sessions on the Uber Eats **mobile-web** site.  
Each simulated run produces a short heuristic-style usability report and score; all runs can be merged into a single, nicely-formatted PDF for human review.

Personas: https://docs.google.com/document/d/1wij_WsHuOH8qoIgsOV3bVFR5y1cv7_NlM5Ua6FRjeUQ/edit?usp=sharing

OS: MacOS<br>
Browser engine: Playwright Chromium<br>
OpenAI account: sk-XXXXXXXXXXXXXXX (Microsoft Team Message needed)<br>
Dependencies: Python 3.11+ | Playwright | OpenAI Python SDK >= 1.0.0 | wkhtmltopdf

# Provided Folder Layout
ueats-llm/<br>
├─ README.md<br>
├─ .gitignore<br>
├─ generate_personas.py    ← generate personas<br>
├─ run_operators.py        ← main script<br>
└─ compose_report.py       ← merges → PDF

# Folder Layout (Ideal)
ueats-llm/<br>
├─ README.md<br>
├─ .gitignore<br>
├─ personas_uniform.json<br>
├─ personas_diet.json<br>
├─ personas_diverse.json<br>
├─ consolidated_report.pdf ← Merged PDF<br>
├─ generate_personas.py    ← generate personas<br>
├─ run_operators.py        ← main script<br>
├─ compose_report.py       ← merges → PDF<br>
├─ runs/                   ← run outcomes<br>
└─ venv/                   ← virtual env

# generate_personas.py
Condition rules:<br>
Uniform: diet=none, accessibility=none, age 19–24, East-Coast US locations, student-appropriate incomes, unique goals.<br>
Diet-only: only diet varies systematically; age 19–30, East-Coast locations, student incomes, accessibility=none, unique goals.<br>
Fully-diverse: everything varies (age, income, worldwide locations, diet, accessibility, goal), with diversity enforced.<br>
Custom Title: python generate_personas.py --condition diet --count 7 --out personas_diet_custom.json

# run_operators.py
1. Open "https://www.ubereats.com" in an iPhone 15 viewport (393×852 px)
2. Feed the page's trimmed DOM (<= 4,000 chars) to GPT-4o-mini along with the active persona
3. Execute the JSON command returned by the model (click, type, wait)
4. Stop at checkout or on error
5. Ask GPT-4o-mini to rate the experience (score: 1.0–5.0), generate description and markdown report
6. Save 'issues.md & issues.json' under 'runs/Condition/Persona_Id/'

# compose_report.py
1. Recursively scan runs/ for issues.json, renders a single HTML summary with Jinja2
2. Convert it to PDF using wkhtmltopdf

# Quick Start
```
# Clone & create a virtual environment (Python 3.11+)
git clone https://github.com/ybaek01/ueats-llm.git
cd ueats-llm
python3.11 -m venv venv && source venv/bin/activate
pip install -U pip
pip install openai playwright pandas rapidfuzz jinja2

# One-time Playwright browser download
playwright install chromium

# Add your OpenAI Key
export OPENAI_API_KEY="sk-XXXXXXXXXXXXX"
```

# One-Time Setup
1. Xcode CLI tools<br>
xcode-select --install

2. Homebrew + core tools<br>
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"<br>
brew install --cask visual-studio-code

3. Install wkhtmltopdf<br>
https://transformy.io/guides/wkhtmltopdf-tutorial/

4. Project + virtual-env<br>
mkdir ~/ueats-llm && cd ~/ueats-llm<br>
python3.11 -m venv venv<br>
source venv/bin/activate<br>
pip install -U pip<br>
pip install openai playwright pandas rapidfuzz jinja2<br>
playwright install chromium

5. OpenAI API Key<br>
echo 'export OPENAI_API_KEY="sk-XXXXXXXXXXXXXXX"' >> ~/.zshrc<br>
source ~/.zshrc<br>
echo $OPENAI_API_KEY ← Check API KEY

6. VS Code Extensions<br>
Python & Playwright Test

# Running the Pipeline in terminal
0) cd ~/ueats-llm<br>
source venv/bin/activate<br>
code .

1) Generate Uniform<br>
python generate_personas.py --condition uniform --count 7

2) Generate Diet-only<br>
python generate_personas.py --condition diet --count 7

3) Generate Fully-diverse<br>
python generate_personas.py --condition diverse --count 7

4) Run Uniform<br>
python run_operators.py --personas personas_uniform.json --output runs/uniform

5) Run Diet-only<br>
python run_operators.py --personas personas_diet.json --output runs/diet

6) Run Fully-diverse<br>
python run_operators.py --personas personas_diverse.json --output runs/diverse

7) Merge PDF<br>
python compose_report.py --input runs --pdf consolidated_report.pdf