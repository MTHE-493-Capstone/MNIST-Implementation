import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def style_axes(ax: plt.Axes, legend_loc: str | None = "best") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if legend_loc:
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc=legend_loc)


def set_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=11)
