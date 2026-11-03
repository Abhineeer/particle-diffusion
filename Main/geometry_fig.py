"""Geometry diagram: the variables, declared before any equation."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14, 7.0))
ax.set_xlim(0, 14.6); ax.set_ylim(-0.2, 7.1); ax.axis("off")

TEAL, AMBER, INK = "#1f6f68", "#b5790f", "#111111"

# trunk
ax.plot([1.6, 6.4], [3.5, 3.5], color=TEAL, lw=11, solid_capstyle="round")
# closed end
ax.plot([1.6, 1.6], [2.65, 4.35], color=INK, lw=5)
ax.text(1.6, 2.1, "closed end", ha="center", fontsize=19, color=INK)

# branches
tips = [(11.4, 6.1), (11.9, 3.5), (11.4, 0.9)]
for x, y in tips:
    ax.plot([6.4, x], [3.5, y], color=TEAL, lw=7, solid_capstyle="round")
    ax.plot(x, y, "o", ms=17, mfc="white", mec=AMBER, mew=4)
ax.text(12.4, 6.1, "uptake", va="center", fontsize=19, color=AMBER)
ax.text(12.9, 3.5, "uptake", va="center", fontsize=19, color=AMBER)
ax.text(12.4, 0.9, "uptake", va="center", fontsize=19, color=AMBER)

# vertex
ax.plot(6.4, 3.5, "o", ms=16, mfc="white", mec=INK, mew=3.5)
ax.text(6.1, 3.85, "junction", ha="right", fontsize=19, color=INK)

# length marks
ax.annotate("", xy=(1.6, 5.45), xytext=(6.4, 5.45),
            arrowprops=dict(arrowstyle="<->", lw=2.2, color=INK))
ax.text(4.0, 5.65, r"$L$", ha="center", fontsize=28, color=INK)
ax.annotate("", xy=(6.55, 4.3), xytext=(11.3, 6.75),
            arrowprops=dict(arrowstyle="<->", lw=2.2, color=INK))
ax.text(8.5, 5.85, r"$L$", ha="center", fontsize=28, color=INK)

# flow arrow, above the trunk
ax.annotate("", xy=(5.4, 4.25), xytext=(2.6, 4.25),
            arrowprops=dict(arrowstyle="-|>,head_width=0.34,head_length=0.7",
                            lw=3.5, color=AMBER))
ax.text(4.0, 4.5, r"$v_0$", ha="center", fontsize=27, color=AMBER)

# variable key
ax.text(0.2, 1.25, r"$L$  vessel length", fontsize=22, color=INK)
ax.text(0.2, 0.68, r"$D$  diffusion constant", fontsize=22, color=INK)
ax.text(5.6, 1.25, r"$b$  branches per junction", fontsize=22, color=INK)
ax.text(5.6, 0.68, r"$v_0$  flow speed in the trunk", fontsize=22, color=INK)

fig.tight_layout()
fig.savefig("fig_geometry.png", dpi=150, facecolor="white")
print("wrote fig_geometry.png")