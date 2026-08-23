import csv
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class ModelStyle:
    label: str
    color: str
    marker: str


MODEL_STYLES = {
    "polymerpim-jit": ModelStyle("Blocking JIT", "#3264a8", "o"),
    "polymerpim-hybrid": ModelStyle("Hybrid", "#dd7f27", "D"),
    "polymerpim-pipeline": ModelStyle("Pipeline", "#3b8f5a", "s"),
    "polymerpim-eager": ModelStyle("Eager", "#8b5aa5", "^"),
    "simplepim": ModelStyle("SimplePIM", "#b94a48", "^"),
}


def load_pyplot(figure):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(f"matplotlib is required to write {figure}: {error}")
    return plt


def write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [field.name for field in fields(rows[0])]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.3f}" if isinstance(value, float) else value
                for key, value in asdict(row).items()
            })


def draw_series(axis, model, points, yerr=None, linewidth=2.2):
    if not points:
        return
    style = MODEL_STYLES[model]
    options = dict(
        color=style.color,
        marker=style.marker,
        linewidth=linewidth,
        markersize=6,
        label=style.label,
    )
    x, y = zip(*points)
    if yerr is None:
        axis.plot(x, y, **options)
    else:
        axis.errorbar(x, y, yerr=yerr, capsize=3, **options)


def configure_axis(axis, ticks, labels, xlabel, minor_grid=False):
    axis.set_xscale("log")
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels)
    axis.set_xlabel(xlabel)
    axis.grid(True, which="major", color="#d8d8d8", linewidth=0.8)
    if minor_grid:
        axis.grid(True, which="minor", color="#eeeeee", linewidth=0.5)


def legend_handles(models, linewidth=2.2):
    from matplotlib.lines import Line2D

    return [Line2D(
        [0], [0], color=MODEL_STYLES[model].color,
        marker=MODEL_STYLES[model].marker, linewidth=linewidth,
        markersize=6, label=MODEL_STYLES[model].label,
    ) for model in models]
