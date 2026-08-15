from real_chart_bench.adapter.starrydata_csv import parse_curve_row


def _row(**overrides):
    base = {
        "SID": "6061",
        "DOI": "10.1000/example",
        "figure_id": "5190",
        "figure_name": "2(a)",
        "prop_x": "Temperature",
        "prop_y": "Seebeck coefficient",
        "unit_x": "K",
        "unit_y": "V*K^(-1)",
        "x": "[299.86,324.87,349.88]",
        "y": "[-0.00014,-0.00016,-0.00017]",
    }
    base.update(overrides)
    return base


def test_parses_json_array_x_y_columns():
    parsed = parse_curve_row(_row())

    assert parsed.sid == "6061"
    assert parsed.figure_id == "5190"
    assert parsed.x_values == (299.86, 324.87, 349.88)
    assert parsed.y_values == (-0.00014, -0.00016, -0.00017)


def test_series_label_combines_prop_and_unit():
    parsed = parse_curve_row(_row())
    assert "Seebeck coefficient" in parsed.series_label


def test_malformed_x_or_y_raises_value_error():
    import pytest

    with pytest.raises(ValueError, match="curve row"):
        parse_curve_row(_row(x="not-json"))


def test_mismatched_length_x_y_raises_value_error():
    import pytest

    with pytest.raises(ValueError, match="length"):
        parse_curve_row(_row(x="[1,2,3]", y="[1,2]"))
