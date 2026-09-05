import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

Path("./results").mkdir(exist_ok=True)
Path("./outputs").mkdir(exist_ok=True)

REGIONS = ["horn_of_africa","ne_brazil","sahel","sw_us","south_asia"]
LABELS  = {"horn_of_africa":"Horn of Africa","ne_brazil":"NE Brazil", "sahel":"Sahel","sw_us":"SW US","south_asia":"South Asia"}
leads   = [1,2,3,4,5,6]

lstm_all, missing = [], []
for r in REGIONS:
    p = f"./results/lstm_{r}.csv"
    if Path(p).exists():
        df = pd.read_csv(p); df["region"] = r; lstm_all.append(df)
        print(f"  Loaded LSTM: {r}")
    else:
        missing.append(r)
        print(f"  Missing: {r} — run lstm_baseline.py first")

if not lstm_all:
    print("No LSTM results found."); exit()

lstm_df = pd.concat(lstm_all, ignore_index=True)

tft_all, pers_all = [], []
for r in REGIONS:
    p = f"./results/ablation_{r}.csv"
    if Path(p).exists():
        df = pd.read_csv(p)
        t = df[df['model']=='TFT-only (no graph)'].copy()
        prs = df[df['model']=='Persistence'].copy()
        t['model'] = 'DroughtTFT'
        t['region'] = r
        prs['region'] = r
        tft_all.append(t); pers_all.append(prs)
    else:
        print(f" ablation file not found for {r}")

if not tft_all:
    print("No DroughtTFT ablation files. Run ablation.py first."); exit()

tft_df   = pd.concat(tft_all,  ignore_index=True)
pers_df  = pd.concat(pers_all, ignore_index=True)
combined = pd.concat([tft_df, lstm_df, pers_df], ignore_index=True)
combined.to_csv("./results/table5_lstm_comparison.csv", index=False)

summary = combined.groupby(['model','Lead (months)'])[['RMSE','R²','POD','FAR']]\
                  .mean().round(3).reset_index()

print("\n" + "="*60)
print("TABLE 5 - Mean Skill: DroughtTFT vs LSTM vs Persistence")
print("(all five regions, test 2018-2023)")
print("="*60)
for model in ['DroughtTFT','LSTM','Persistence']:
    sub = summary[summary['model']==model]
    if len(sub):
        print(f"\n{model}:")
        print(sub[['Lead (months)','RMSE','R²','POD']].to_string(index=False))

tft_s  = summary[summary['model']=='DroughtTFT'].set_index('Lead (months)')
lstm_s = summary[summary['model']=='LSTM'].set_index('Lead (months)')
pers_s = summary[summary['model']=='Persistence'].set_index('Lead (months)')

print("\n" + "="*60)
print("DELTA R² = DroughtTFT - LSTM  (positive = TFT better)")
print("="*60)
for l in leads:
    if l in tft_s.index and l in lstm_s.index:
        d = tft_s.loc[l,'R²'] - lstm_s.loc[l,'R²']
        print(f"  Lead {l}: {d:+.3f}  "
              f"({'TFT better' if d>0 else 'LSTM better or equal'})")

plt.rcParams.update({'font.family':'serif','font.size':11, 'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(1,2,figsize=(12,5))

ax = axes[0]
STYLES = {
    'DroughtTFT': dict(color='#1d4ed8',marker='o',lw=2.5,ms=8,ls='-',  alpha=1.0), 'LSTM': dict(color='#dc2626',marker='s',lw=2.5,ms=8,ls='-',  alpha=1.0), 'Persistence': dict(color='#6b7280',marker='D',lw=1.8,ms=6,ls='--', alpha=0.75),
}
for model in ['DroughtTFT','LSTM','Persistence']:
    sub = summary[summary['model']==model]
    if len(sub):
        ax.plot(sub['Lead (months)'],sub['R²'],label=model,**STYLES[model])

ax.axhline(0,color='#374151',lw=0.9,ls=':',zorder=2)
ax.fill_between([0.6,6.4], 0, 0.65,alpha=0.04,color='#22c55e')
ax.fill_between([0.6,6.4],-0.3,0,  alpha=0.05,color='#ef4444')
ax.set_xlim(0.6,6.4); ax.set_ylim(-0.30,0.68)
ax.set_xlabel('Lead Time (months)',fontsize=12)
ax.set_ylabel('Mean R² (SPEI-6)',fontsize=12)
ax.set_title('(a) Mean R²: DroughtTFT vs. LSTM vs. Persistence\n' '(mean across all 5 regions)',fontsize=11,fontweight='bold')
ax.set_xticks(leads)
ax.grid(True,alpha=0.18)
ax.legend(fontsize=10,framealpha=0.95,edgecolor='#d1d5db')

ax2 = axes[1]
RCOLS = ['#1d4ed8','#15803d','#b45309','#7c3aed','#dc2626']
for i,region in enumerate(REGIONS):
    tr = tft_df[tft_df['region']==region].set_index('Lead (months)')
    lr = lstm_df[lstm_df['region']==region].set_index('Lead (months)')
    if len(tr) and len(lr):
        delta = [tr.loc[l,'R²']-lr.loc[l,'R²']
                 if l in tr.index and l in lr.index else np.nan
                 for l in leads]
        ax2.plot(leads,delta,'o-',color=RCOLS[i],lw=2.0,ms=6.5, label=LABELS[region])

ax2.axhline(0,color='#374151',lw=1.2,zorder=2)
ax2.fill_between([0.6,6.4], 0, 0.30,alpha=0.05,color='#1d4ed8')
ax2.fill_between([0.6,6.4],-0.25,0, alpha=0.05,color='#dc2626')
ax2.set_xlim(0.6,6.4)
ax2.set_xlabel('Lead Time (months)',fontsize=12)
ax2.set_ylabel('ΔR² = DroughtTFT − LSTM',fontsize=12)
ax2.set_title('(b) TFT Advantage over LSTM by Region\n' '(positive = TFT better)',fontsize=11,fontweight='bold')
ax2.set_xticks(leads)
ax2.grid(True,alpha=0.18)
ax2.legend(fontsize=9.5,framealpha=0.95,edgecolor='#d1d5db')
ax2.text(1.0, 0.008,'TFT better', fontsize=8.5,color='#1d4ed8',alpha=0.8)
ax2.text(1.0,-0.22, 'LSTM better',fontsize=8.5,color='#dc2626',alpha=0.7)

plt.suptitle('DroughtTFT vs. LSTM Baseline\n(ERA5, Test Period 2018–2023)', fontsize=12.5,fontweight='bold',y=1.01)
plt.tight_layout()
for ext in ['pdf','png']:
    fig.savefig(f'./outputs/figure5_lstm_comparison.{ext}', dpi=300,bbox_inches='tight')
plt.close()
print("Figure 5 saved.")

def fmt(val):
    if pd.isna(val): return '--'
    s = f"{abs(val):.3f}"
    return s if val >= 0 else f"$-${s}"

def best_in_col(models, df, lead, metric, want_max):
    vals = {m: df[df['model']==m].set_index('Lead (months)').loc[lead,metric]
            for m in models
            if lead in df[df['model']==m]['Lead (months)'].values}
    if not vals: return None
    return max(vals,key=vals.get) if want_max else min(vals,key=vals.get)

models = ['DroughtTFT','LSTM','Persistence']
latex = [
    r"\begin{table}[h]",
    r"\caption{Mean test-set skill across all five regions: DroughtTFT, LSTM",
    r"baseline, and anomaly persistence (test period 2018--2023).",
    r"RMSE in SPEI-6 units; mean column averages leads 1--6.",
    r"Best value per column in \textbf{bold}.}",
    r"\label{tab:lstm}",
    r"\centering",r"\small",
    r"\begin{tabular}{l|cc|cc|cc|cc}",
    r"\toprule",
    r" & \multicolumn{2}{c|}{Lead 1} & \multicolumn{2}{c|}{Lead 3}"
    r" & \multicolumn{2}{c|}{Lead 6} & \multicolumn{2}{c}{Mean (leads 1--6)} \\",
    r"Model & RMSE & $R^2$ & RMSE & $R^2$ & RMSE & $R^2$ & RMSE & $R^2$ \\",
    r"\midrule",
]
for model in models:
    ds   = summary[summary['model']==model].set_index('Lead (months)')
    row  = [f"\\textbf{{{model}}}" if model=='DroughtTFT' else model]
    mRMSE, mR2 = [], []
    for l in [1,3,6]:
        if l in ds.index:
            rmse = ds.loc[l,'RMSE']; r2 = ds.loc[l,'R²']
            mRMSE.append(rmse); mR2.append(r2)
            br = best_in_col(models,summary,l,'RMSE',False)
            b2 = best_in_col(models,summary,l,'R²',True)
            rs  = f"\\textbf{{{rmse:.3f}}}" if br==model else f"{rmse:.3f}"
            r2s = f"\\textbf{{{fmt(r2)}}}" if b2==model else fmt(r2)
            row += [rs, r2s]
        else:
            row += ['--','--']
    mr   = np.mean(mRMSE) if mRMSE else np.nan
    mr2  = np.mean(mR2)   if mR2   else np.nan
    bMR  = min({m: summary[summary['model']==m]['RMSE'].mean() for m in models}, key=lambda m: summary[summary['model']==m]['RMSE'].mean())
    bMR2 = max({m: summary[summary['model']==m]['R²'].mean()   for m in models}, key=lambda m: summary[summary['model']==m]['R²'].mean())
    mrs  = f"\\textbf{{{mr:.3f}}}" if bMR==model else f"{mr:.3f}"
    mr2s = f"\\textbf{{{fmt(mr2)}}}" if bMR2==model else fmt(mr2)
    row += [mrs, mr2s]
    latex.append(" & ".join(row) + r" \\")

latex += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
with open("./results/table5_lstm_latex.txt","w") as f:
    f.write("\n".join(latex))
print("Saved: ./results/table5_lstm_latex.txt")
print("Saved: ./results/table5_lstm_comparison.csv")

if missing:
    print(f"\nStill need LSTM for: {missing}")
else:
    print("\nAll 5 regions complete. Table 5 and Figure 5 ready for paper.")
