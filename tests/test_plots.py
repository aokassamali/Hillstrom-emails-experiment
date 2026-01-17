from pathlib import Path
import tempfile

import pandas as pd

from plots import plot_arm_sizes, plot_outcome_means_ci, plot_uplift_ci


def test_plots_write_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        arm_sizes = pd.DataFrame({"arm": ["control", "mens"], "count": [10, 12]})
        plot_arm_sizes(arm_sizes, tmp / "arm_sizes.png")
        assert (tmp / "arm_sizes.png").exists()

        outcome_means = pd.DataFrame(
            {
                "outcome": ["profit", "profit"],
                "arm": ["control", "mens"],
                "mean": [0.0, 0.1],
                "ci_low": [-0.1, 0.0],
                "ci_high": [0.1, 0.2],
            }
        )
        plot_outcome_means_ci(outcome_means, tmp)
        assert (tmp / "mean_ci_profit.png").exists()

        uplift = pd.DataFrame(
            {
                "outcome": ["profit"],
                "arm": ["mens"],
                "estimate": [0.1],
                "ci_low": [0.0],
                "ci_high": [0.2],
            }
        )
        plot_uplift_ci(uplift, tmp)
        assert (tmp / "uplift_ci_profit.png").exists()
