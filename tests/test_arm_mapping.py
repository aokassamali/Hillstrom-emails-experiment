from hillstrom_emails.cleaning import normalize_arm


def test_arm_mapping():
    arm_map = {
        "control": ["No E-Mail"],
        "mens": ["Mens E-Mail"],
        "womens": ["Womens E-Mail"],
    }
    assert normalize_arm("Mens E-Mail", arm_map) == "mens"
    assert normalize_arm("Womens E-Mail", arm_map) == "womens"
    assert normalize_arm("No E-Mail", arm_map) == "control"
    assert normalize_arm("unknown", arm_map) is None
