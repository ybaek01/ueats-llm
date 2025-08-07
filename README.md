# Usability-Testing-2
Yoongi Baek



# 1) Uniform
python run_operators.py --personas personas_uniform.json --output runs/uniform

# 2) Diet-only
python run_operators.py --personas personas_diet.json --output runs/diet

# 3) Fully-diverse
python run_operators.py --personas personas_diverse.json --output runs/diverse

# 4) Combine PDF
python compose_report.py --input runs --pdf consolidated_report.pdf