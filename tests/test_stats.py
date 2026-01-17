import numpy as np
from hillstrom_emails.stats import holm_adjust


def test_holm_adjust():
    p = np.array([0.01, 0.04])
    adj = holm_adjust(p)
    assert np.allclose(adj, np.array([0.02, 0.04]))
