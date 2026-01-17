from hillstrom_emails.config import load_config


def test_srm_expected_allocation_default():
    cfg = load_config("config/config.yaml")
    expected = cfg["srm"].get("expected_allocation", {})
    assert round(expected.get("control", 0), 6) == round(1 / 3, 6)
    assert round(expected.get("mens", 0), 6) == round(1 / 3, 6)
    assert round(expected.get("womens", 0), 6) == round(1 / 3, 6)
