# Persona Diversity Simulation
Persona Diversity: Comparing Usability-Testing Outcomes across Heterogeneous Personas

A lightweight pipeline that combines Playwright, OpenAI GPT-4o-mini, and JSON persona files to simulate end-to-end food-ordering sessions on the Uber Eats **mobile-web** site.  
Each simulated run produces a short heuristic-style usability report; all runs can be merged into a single, nicely-formatted PDF for human review.

Personas: https://docs.google.com/document/d/1wij_WsHuOH8qoIgsOV3bVFR5y1cv7_NlM5Ua6FRjeUQ/edit?usp=sharing

OS: MacOS<br>
Browser engine: Playwright Chromium<br>
OpenAI account: sk-XXXXXXXXXXXXXXX (Microsoft Team Message needed)<br>
Dependencies: Python 3.11+ | Playwright | OpenAI Python SDK >= 1.0.0 | wkhtmltopdf

# Folder Layout
ueats-llm/<br>
├─ personas_uniform.json<br>
├─ personas_diet.json<br>
├─ personas_diverse.json<br>
├─ run_operators.py      ← main script<br>
├─ compose_report.py     ← merges → PDF<br>
└─ venv/                 ← virtual env<br>

# run_operators.py
1. Open "https://www.ubereats.com" in an iPhone 15 viewport (393×852 px)
2. Feed the page's trimmed DOM (<= 4,000 chars) to GPT-4o-mini along with the active persona
3. Execute the JSON command returned by the model (click, type, wait)
4. Stop at checkout or on error
5. Save 'issues.md & issues.json' under 'runs/condition/id/'

# compose_report.py
1. Recursively scan runs/ for issues.json, renders a single HTML summary with Jinja2
2. Convert it to PDf using wkhtmltopdf

# One-Time Setup
1. Xcode CLI tools<br>
xcode-select --install

2. Homebrew + core tools<br>
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"<br>
brew install --cask visual-studio-code<br>

3. Install wkhtmltopdf<br>
https://transformy.io/guides/wkhtmltopdf-tutorial/

4. Project + virtual-env<br>
mkdir ~/ueats-llm && cd ~/ueats-llm<br>
python3.11 -m venv venv<br>
source venv/bin/activate<br>
pip install -U pip<br>
pip install openai playwright pandas rapidfuzz jinja2<Br>
playwright install chromium<Br>

5. OpenAI API Key<br>
echo 'export OPENAI_API_KEY="sk-XXXXXXXXXXXXXXX"' >> ~/.zshrc
source ~/.zshrc

6. VS Code Extensions<br>
Python & Playwright Test

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

# Running the Pipeline in terminal
0) cd ~/ueats-llm<br>
source venv/bin/activate<br>
code .<br>


1) Uniform<br>
python run_operators.py --personas personas_uniform.json --output runs/uniform

2) Diet-only<br>
python run_operators.py --personas personas_diet.json --output runs/diet

3) Fully-diverse<br>
python run_operators.py --personas personas_diverse.json --output runs/diverse

4) Merge PDF<br>
python compose_report.py --input runs --pdf consolidated_report.pdf