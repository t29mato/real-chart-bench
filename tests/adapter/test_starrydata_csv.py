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


def test_composition_and_sample_id_are_carried_through():
    """The published schema has `composition` and `sample_id`; they are the only
    thing that distinguishes curves within a figure (prop_y is identical across
    all of them), so the parser must not drop them.
    See docs/experiments/2026-10-07-series-labels-available.md."""
    row = {
        "SID": "1",
        "DOI": "10.1/x",
        "composition": "Pb1.00025Zn0.02Te1.02I0.0005",
        "sample_id": "4242",
        "figure_id": "79",
        "figure_name": "Fig. 3",
        "prop_x": "Temperature",
        "prop_y": "Seebeck coefficient",
        "unit_x": "K",
        "unit_y": "V/K",
        "x": "[300.0, 400.0]",
        "y": "[1.0, 2.0]",
    }
    parsed = parse_curve_row(row)
    assert parsed.composition == "Pb1.00025Zn0.02Te1.02I0.0005"
    assert parsed.sample_id == "4242"
    # the geometric label stays what it was -- scoring must not shift
    assert parsed.series_label == "Seebeck coefficient (V/K)"


def test_missing_composition_and_sample_id_become_empty_strings():
    row = {
        "SID": "1",
        "DOI": "",
        "figure_id": "79",
        "prop_y": "Seebeck coefficient",
        "unit_y": "",
        "x": "[1.0]",
        "y": "[2.0]",
    }
    parsed = parse_curve_row(row)
    assert parsed.composition == ""
    assert parsed.sample_id == ""
