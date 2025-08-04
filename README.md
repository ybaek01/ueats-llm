# Usability-Testing-2
Yoongi Baek

Persona Diversity: Comparing Usability-Testing Outcomes across Heterogeneous Personas

Here for personas: https://docs.google.com/document/d/1wij_WsHuOH8qoIgsOV3bVFR5y1cv7_NlM5Ua6FRjeUQ/edit?usp=sharing 

Also, you have to install openAI jinja 2 playwright from chromium

If you guys have any issues, message me.


** Since we are using openAI tokens, it costs. Probably less than $ 1 per one simulation. **
Playwright is free, also a jinja2.

I have credit on openAI, so it does not matter.

Also, it would take 20-30 minutes per one condition.
At most 1.5 hr for 3 condition: Uniform, Diet-only, Fully-Diverse.


Final Running would be below...

# 1) Uniform
python run_operators.py --personas personas_uniform.json --output runs/uniform

# 2) Diet-only
python run_operators.py --personas personas_diet.json --output runs/diet

# 3) Fully-diverse
python run_operators.py --personas personas_diverse.json --output runs/diverse

# 4) Combine PDF
python compose_report.py --input runs --pdf consolidated_report.pdf