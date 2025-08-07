# Usability-Testing-2
Yoongi Baek

Persona Diversity: Comparing Usability-Testing Outcomes across Heterogeneous Personas

Personas: https://docs.google.com/document/d/1wij_WsHuOH8qoIgsOV3bVFR5y1cv7_NlM5Ua6FRjeUQ/edit?usp=sharing

OS: MacOS<br>
Browser engine: Playwright Chromium<br>
OpenAI account: sk-XXXXXXXXXXXXXXX (Microsoft Team Message needed)


# One-Time Setup
1. Xcode CLI tools<br>
xcode-select --install

2. Homebrew + core tools<br>
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"<br>
brew install git python@3.11 wkhtmltopdf<br>
brew install --cask visual-studio-code<br>

3. Project + virtual-env<br>
mkdir ~/ueats-llm && cd ~/ueats-llm<br>
python3.11 -m venv venv<br>
source venv/bin/activate<br>
pip install -U pip<br>
pip install openai playwright pandas rapidfuzz jinja2<Br>
playwright install chromium<Br>

4. OpenAI API Key<br>
echo 'export OPENAI_API_KEY="sk-XXXXXXXXXXXXXXX"' >> ~/.zshrc
source ~/.zshrc

5. VS Code Extensions<br>
Python & Playwright Test


# Folder Layout
ueats-llm/
├─ personas_uniform.json
├─ personas_diet.json
├─ personas_diverse.json
├─ run_operators.py      ← main script
├─ compose_report.py     ← merges → PDF
└─ venv/                 ← virtual env

# run_operators.py
Opens Uber Eats mobile-web in an iPhone 14 Pro viewport

Sets the delivery address (100 7th Ave, NYC)

Searches Buffalo Wild Wings → adds Buffalo Wings to cart

Captures final review screen (text + HTML)

Sends it to GPT-4o-mini → extracts heuristic issues

Saves to runs/<condition>/<ID>/issues.json


# Running the Pipeline in terminal
0) cd ~/ueats-llm<br>
source venv/bin/activate<br>
code .<br>

1) Uniform
python run_operators.py --personas personas_uniform.json --output runs/uniform

2) Diet-only
python run_operators.py --personas personas_diet.json --output runs/diet

3) Fully-diverse
python run_operators.py --personas personas_diverse.json --output runs/diverse

4) Merge PDF
python compose_report.py --input runs --pdf consolidated_report.pdf