import os
import pandas as pd
import matplotlib.pyplot as plt



# ========================== with cueq on NV ================
# only forward
df_time = pd.DataFrame(
    {
        "other": [1.0,  0.7, 1.4, 4.2, 6.4],
        "cwtp": [1.13, 2.21, 25.0, 9.78, 8.79],
        "fctp": [1.20,  1.37,  2.7, 3.25, 0],
        "stc":  [2.04,  4.23, 8.5, 7.02, 0],
        "eq-linear": [0.91, 1.35, 2.1, 12.81, 13.4],
        
    },

    index=["MACE-OFF small", "MACE-OFF medium", "MACE-OFF large", "MACE-MP-0 medium", "SevenNet-0"],
)

df_pct = df_time.div(df_time.sum(axis=1), axis=0) * 100
df_pct = df_pct.drop(columns=["other"], errors="ignore")

cols = list(df_pct.columns)

fig, ax = plt.subplots(figsize=(9, 5))
df_pct.plot(kind="bar", stacked=True, ax=ax)

ax.set_ylabel("Share of total inference time (%)")
ax.set_xlabel("Model size (MACE-OFF)")
ax.set_ylim(0, 100)
ax.set_title("MACE-OFF/SevenNet-0 operator time breakdown with cuEquivariance on Nvidia H100")
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")


for i, container in enumerate(ax.containers):
    #texts = ax.bar_label(container, fmt="%.1f%%", label_type="center", fontsize=9, padding=0)
    vals = container.datavalues
    col = cols[i] 
    if col == "other":
        labels = [""] * len(vals)  # other 不显示任何百分比
    else:
        labels = [f"{v:.1f}%" if v >= 0.1 else "" for v in vals]
    texts = ax.bar_label(container, labels=labels, label_type="center", fontsize=9, padding=0)
    for t in texts:
        t.set_clip_on(True)

leg = ax.legend(
    #title="Operator",
    loc="upper right",            
    bbox_to_anchor=(0.98, 0.98),  
    borderaxespad=0.0,
    frameon=True
)
plt.tight_layout()

out_path = "./nv_cueq_stacked_bar.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)

(out_path, os.path.exists(out_path), os.path.getsize(out_path) if os.path.exists(out_path) else None)


# ========================== with openeq on HYGON ================

df_time = pd.DataFrame(
    {
        "other": [1.5,  4,  6],
        "cwtp": [6.5, 14.1, 40.4],
        "fctp": [1.8,  2.7,  6.4],
        "stc":  [15.7,  54.4, 168],
        "eq-linear": [1.6, 2.3, 5.0],
        
    },
    index=["MACE-OFF small", "MACE-OFF medium", "MACE-OFF large"],
)

df_pct = df_time.div(df_time.sum(axis=1), axis=0) * 100
df_pct = df_pct.drop(columns=["other"], errors="ignore")


cols = list(df_pct.columns)

fig, ax = plt.subplots(figsize=(9, 5))
df_pct.plot(kind="bar", stacked=True, ax=ax)

ax.set_ylabel("Share of total inference time (%)")
ax.set_xlabel("Model size (MACE-OFF)")
ax.set_ylim(0, 100)
ax.set_title("MACE-OFF operator time breakdown with openEquivariance on Hygon BW200")
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

for i, container in enumerate(ax.containers):
    #texts = ax.bar_label(container, fmt="%.1f%%", label_type="center", fontsize=9, padding=0)
    vals = container.datavalues
    col = cols[i]
    if col == "other":
        labels = [""] * len(vals)  # other 不显示任何百分比
    else:
        labels = [f"{v:.1f}%" if v >= 0.1 else "" for v in vals]
    texts = ax.bar_label(container, labels=labels, label_type="center", fontsize=9, padding=0)
    for t in texts:
        t.set_clip_on(True)

leg = ax.legend(
    loc="upper right",            
    bbox_to_anchor=(0.98, 0.98),  
    borderaxespad=0.0,
    frameon=True
)
plt.tight_layout()

out_path = "./hy_oeq_stacked_bar.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)

(out_path, os.path.exists(out_path), os.path.getsize(out_path) if os.path.exists(out_path) else None)