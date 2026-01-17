from __future__ import annotations

import platform
import numpy as np
import pandas as pd
import scipy
import statsmodels


def main() -> None:
    print(f"Python: {platform.python_version()}")
    print(f"numpy: {np.__version__}")
    print(f"pandas: {pd.__version__}")
    print(f"scipy: {scipy.__version__}")
    print(f"statsmodels: {statsmodels.__version__}")

    df = pd.DataFrame({
        "arm": ["control", "mens", "womens"],
        "emailed": [0, 1, 1],
        "spend": [0.0, 10.0, 5.0],
    })
    profit = 0.40 * df["spend"] - 0.01 * df["emailed"]
    assert profit.shape[0] == 3
    print("Doctor: smoke test passed")


if __name__ == "__main__":
    main()
