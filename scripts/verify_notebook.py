"""Verificar la estructura del notebook generado."""
import json

with open("notebooks_val/0_Autopsia_Motor_KineTopus.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Formato: nbformat {nb['nbformat']}")
print(f"Celdas totales: {len(nb['cells'])}")
md_count = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')
code_count = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
print(f"  Markdown: {md_count}")
print(f"  Code:     {code_count}")
print()
for i, c in enumerate(nb['cells']):
    first_line = c['source'][0].strip()[:80] if c['source'] else "(vacia)"
    print(f"  [{i:2d}] {c['cell_type']:8s} | {first_line}")
