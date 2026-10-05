"""Chart-to-table models (design §7.68, §7.75) answer with a text table. The
parser turns it into series without guessing: first column is x, every other
column a series, rows whose x is not a number are dropped (headers, axis
titles the model copied in), exact repeats are dropped, nothing is
interpolated. A table with no numeric x at all is unparseable -- scored as no
answer, and counted separately."""

import pytest

from real_chart_bench.adapter.chart_table_parser import parse_chart_table, parse_number

DEPLOT = (
    "TITLE |  <0x0A> Seebeck (μV/K) | 1150 °C | 1200 °C <0x0A> Temperature (°C) | -10.5 | 48.8 "
    "<0x0A> 600 | -76.1 | 47.1 <0x0A> 650 | -62.2 | 52.0 <0x0A> 650 | -62.2 | 52.0"
)


def test_deplot_table_becomes_one_series_per_column():
    table = parse_chart_table(DEPLOT)

    assert table.parsed
    assert [s.label for s in table.series] == ["1150 °C", "1200 °C"]
    assert table.series[0].x == (600.0, 650.0)
    assert table.series[0].y == (-76.1, -62.2)
    assert table.series[1].y == (47.1, 52.0)
    assert table.n_rows_dropped == 1  # "Temperature (°C)" row


def test_unichart_rows_are_separated_by_ampersand():
    text = "Year | A | B & 1 | 2 | 3 & 2 | 4 | 6"

    table = parse_chart_table(text, row_separator=" & ")

    assert [s.label for s in table.series] == ["A", "B"]
    assert table.series[1].y == (3.0, 6.0)


def test_csv_with_header():
    text = "T (K),x=0.1,x=0.2\n300,1.5,2.5\n400,1.7,2.9\n"

    table = parse_chart_table(text)

    assert [s.label for s in table.series] == ["x=0.1", "x=0.2"]
    assert table.series[0].x == (300.0, 400.0)


def test_markdown_table_separator_rows_are_ignored():
    text = "| x | s |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"

    table = parse_chart_table(text)

    assert table.series[0].x == (1.0, 3.0)
    assert table.series[0].y == (2.0, 4.0)


def test_a_missing_cell_leaves_that_point_out_of_its_series_only():
    text = "x | a | b\n1 | 2 | \n3 | 4 | 5"

    table = parse_chart_table(text)

    assert table.series[0].x == (1.0, 3.0)
    assert table.series[1].x == (3.0,)


def test_no_numeric_x_is_unparseable():
    text = "Country | GDP\nJapan | 4.2\nFrance | 2.9"

    table = parse_chart_table(text)

    assert not table.parsed
    assert table.series == ()


def test_empty_and_none_are_unparseable():
    assert not parse_chart_table("").parsed
    assert not parse_chart_table(None).parsed


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("12", 12.0),
        ("-3.5", -3.5),
        ("−3.5", -3.5),  # unicode minus
        ("1,014", 1014.0),
        ("12%", 12.0),
        ("1.2e-5", 1.2e-5),
        ("1.2×10^-5", 1.2e-5),
        ("3.0x10^4", 3.0e4),
        ("10^3", 1e3),
        ("abc", None),
        ("", None),
        ("nan", None),
    ],
)
def test_parse_number(text, value):
    got = parse_number(text)
    if value is None:
        assert got is None
    else:
        assert got == pytest.approx(value)
