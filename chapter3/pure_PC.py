import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def panel_label(ax, text):
    ax.text(-0.12, 1.06, text, transform=ax.transAxes,
            ha="left", va="bottom", fontweight="bold", fontsize=28)

# ---------- вспомогательные ----------
def piecewise_stim(t):
    if t < 0.2:  return 0.0
    if t < 3.0:  return 1.0
    if t < 10.0: return -1.0
    return 0.5

# ---------- МОДЕЛЬ 1: error-driven integrator ----------
T1, dt1, k = 15.0, 0.001, 0.7
t1 = np.arange(0.0, T1 + dt1, dt1) # (15001,) [0-15]
l1 = np.array([piecewise_stim(tt) for tt in t1]) # (15001,)

l2 = np.zeros_like(t1)  # "2 популяция" (15001,)
e = np.zeros_like(t1)  # ошибка (15001,)
for n in range(1, len(t1)):
    e[n-1] = k * (l1[n-1] - l2[n-1])
    l2[n]   = l2[n-1] + dt1 * e[n-1]
e[-1] = k * (l1[-1] - l2[-1])

# ---------- графика 2×2 ----------
plt.figure()
plt.plot(t1, e,    label="Ошибка",      linestyle="--", linewidth=2.6)
plt.plot(t1, l1, label="1 популяция", linestyle="-",  linewidth=2.6)
plt.plot(t1, l2,    label="2 популяция", linestyle=":",  linewidth=2.6)
plt.title("Выходные сигналы популяций")
plt.xlabel("Время (сек)"); 
plt.ylabel("Выходное значение")
plt.legend(frameon=False); 
plt.grid(True, which="major", linestyle="--", alpha=0.35)
plt.minorticks_on(); 
plt.grid(True, which="minor", linestyle=":", alpha=0.2)
plt.xlim(t1[0], t1[-1]); 
overview_path = "overview_PC.png"
plt.savefig(overview_path, dpi=200, bbox_inches="tight")
plt.show()