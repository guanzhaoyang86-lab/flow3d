"""Export existing research figures without presentation prose.

Numerical data and scientific labels are unchanged. The original report
figures remain untouched; pass a separate --output-dir for these assets.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import visualize_particle_count_results as source

SELECTED = {
    "00_pipeline_overview",
    "01_flow_slices_median",
    "04b_track_error_over_time_median",
    "05b_aggregate_metrics_identity",
    "06_uncertainty_vs_error_median",
    "07_training_convergence_and_loss_scale",
    "08_flow_field_3d_median",
    "09_best_median_worst_gallery",
}
_original_save = source._save


def presentation_save(fig, output_dir: Path, stem: str, dpi: int):
    if stem not in SELECTED:
        plt.close(fig)
        return []
    if fig._suptitle is not None:
        fig._suptitle.remove()
        fig._suptitle = None
    remove_prefixes = (
        "Ground truth is a training target",
        "Sparse multi-view particle tracks",
        "Single training seed",
        "Explicit track and physics terms",
    )
    for ax in fig.axes:
        for artist in list(ax.texts):
            if artist.get_text().startswith(remove_prefixes):
                artist.remove()
    if stem == "00_pipeline_overview":
        fig.axes[0].set_ylim(0.23, 0.88)
    if stem == "05b_aggregate_metrics_identity":
        handles = [
            Line2D([], [], marker="o", linestyle="none", color="#0072B2",
                   label="N=2 better"),
            Line2D([], [], marker="o", linestyle="none", color="#D55E00",
                   label="N=32 better"),
            Line2D([], [], linestyle="--", color="#555555", label="Equal"),
        ]
        fig.legend(handles=handles, loc="upper center", ncol=3,
                   bbox_to_anchor=(0.5, 1.035), fontsize=10)
    return _original_save(fig, output_dir, stem, dpi)


if __name__ == "__main__":
    source._save = presentation_save
    source.main()
