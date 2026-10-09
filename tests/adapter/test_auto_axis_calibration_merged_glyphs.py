"""v3 order fix (docs/design/local-model.md「自動校正の改善」): tick labels that
Tesseract reads as one word ("300320340...") are split into labels BEFORE the
dropped decimal point / minus sign is restored from the ink. Restoring first
put a point inside the merged word ("300320340.36...") so it could no
longer be split, and the x axis was lost (benchmark 28331, v3 only)."""

from __future__ import annotations

from real_chart_bench.adapter import auto_axis_calibration as A

from .test_auto_axis_calibration_dual_y import X_TICKS, _ocr, _rgb


def _merged_ocr(strip, psm=11, **kw):
    if strip.shape[1] > strip.shape[0] and strip.shape[1] > 200:  # x strip: one word
        sx0 = 7
        text = "".join(str(300 + 20 * k) for k in range(11))
        step = X_TICKS[1] - X_TICKS[0]  # one label per tick spacing, centred on its tick
        a = X_TICKS[0] - sx0 - step / 2
        return [(text, a, 5.0, a + step * len(X_TICKS), 15.0, 90.0)]
    return _ocr(strip, psm, **kw)


def _gapped_spans(ink, n, **_):
    """The labels' ink separated by blank gaps, as in a real figure (the test
    image draws no label ink)."""
    step = ink.shape[1] / n
    return [(round(k * step + 3), round((k + 1) * step - 3)) for k in range(n)]


def test_merged_x_labels_survive_a_glyph_reading_a_point(monkeypatch):
    monkeypatch.setattr(A, "ocr_words", _merged_ocr)
    monkeypatch.setattr(A, "split_at_widest_gaps", _gapped_spans)
    # the ink shows a point after the third digit (as for "5.0" labels)
    monkeypatch.setattr(A, "label_glyphs", lambda _crop: {"dot_after": 3})
    c = A.calibrate_image(_rgb(), v3=True, right_axes=False)
    assert c is not None and c.x_fit is not None
    assert abs(c.x_fit.px_to_value(X_TICKS[5]) - 400) < 1


def test_v2_reads_merged_labels_as_before(monkeypatch):
    monkeypatch.setattr(A, "ocr_words", _merged_ocr)
    monkeypatch.setattr(A, "split_at_widest_gaps", _gapped_spans)
    c = A.calibrate_image(_rgb(), right_axes=False)
    assert c is not None and c.x_fit is not None
