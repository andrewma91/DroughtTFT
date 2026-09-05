import pandas as pd
import numpy as np
from pathlib import Path

Path("./results").mkdir(exist_ok=True)

REGIONS = ["horn_of_africa", "ne_brazil", "sahel", "sw_us", "south_asia"]
MODELS  = ["DroughtGAT (full)", "TFT-only (no graph)", "Persistence", "climatology"]

all_dfs = []
missing = []
for region in REGIONS:
    path = f"./results/ablation_{region}.csv"
    if Path(path).exists():
        df = pd.read_csv(path)
        all_dfs.append(df)
        print(f"Loaded: {region}")

if not all_dfs:
    print("No ablation results found.")
    exit()

combined = pd.concat(all_dfs, ignore_index=True)


print("\n" + "="*70)
print("="*70)

for model in MODELS:
    sub = summary[summary["model"] == model]
    if len(sub):
        print(f"\n{model}:")
        print(sub[["Lead (months)","RMSE","R²","POD"]].to_string(index=False))

gat  = "DroughtGAT (full)"].set_index("Lead (months)")
tft  = "TFT-only (no graph)"].set_index("Lead (months)")
pers = "Persistence"].set_index("Lead (months)")
clim = "Climatology"].set_index("Lead (months)")

print("\n" + "="*70)
print("="*70)
for lead in sorted(gat.index):
    delta = gat.loc[lead, "R²"] - tft.loc[lead, "R²"]
    print(f"  Lead {lead}mo: ΔR² = {delta:+.3f} "
          f"({'graph helps' if delta > 0 else 'graph hurts'})")

latex = [
    r"\begin{table}[ht]",
    r"\centering",
    r"\caption{Ablation study: mean test-set skill across all five regions."
    r" RMSE in SPEI-6 units. Best value per lead time in bold.}",
    r"\label{tab:ablation}",
    r"\begin{tabular}{l|cc|cc|cc|cc}",
    r"\hline",
    r" & \multicolumn{2}{c|}{Lead 1} & \multicolumn{2}{c|}{Lead 3}"
    r" & \multicolumn{2}{c|}{Lead 6} & \multicolumn{2}{c}{Mean} \\",
    r"Model & RMSE & R$^2$ & RMSE & R$^2$ & RMSE & R$^2$ & RMSE & R$^2$ \\",
    r"\hline",
]

for model in MODELS:
    sub = summary[summary["model"] == model].set_index("Lead (months)")
    if len(sub) == 0:
        continue
    def get(lead, col):
        return f"{sub.loc[lead, col]:.3f}" if lead in sub.index else "--"
    mean_rmse = sub["RMSE"].mean()
    mean_r2   = sub["R²"].mean()
    row = (f"{model} & {get(1,'RMSE')} & {get(1,'R²')} & "
           f"{get(3,'RMSE')} & {get(3,'R²')} & "
           f"{get(6,'RMSE')} & {get(6,'R²')} & "
           f"{mean_rmse:.3f} & {mean_r2:.3f} \\\\")
    latex.append(row)

latex += [r"\hline", r"\end{tabular}", r"\end{table}"]
latex_str = "\n".join(latex)

with open("./results/table3_ablation_latex.txt", "w") as f:
    f.write(latex_str)
