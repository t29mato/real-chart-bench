import pytest

from real_chart_bench.domain.curve import ScaleType
from real_chart_bench.domain.pixel_calibration import PixelCalibration


def test_maps_top_left_pixel_to_x_min_y_max():
    calib = PixelCalibration(pixel_bbox=(0, 0, 100, 200), x_range=(0, 10), y_range=(0, 20))
    x, y = calib.to_data(0, 0)
    assert x == pytest.approx(0)
    assert y == pytest.approx(20)


def test_maps_bottom_right_pixel_to_x_max_y_min():
    calib = PixelCalibration(pixel_bbox=(0, 0, 100, 200), x_range=(0, 10), y_range=(0, 20))
    x, y = calib.to_data(100, 200)
    assert x == pytest.approx(10)
    assert y == pytest.approx(0)


def test_maps_center_pixel_to_midpoint():
    calib = PixelCalibration(pixel_bbox=(0, 0, 100, 100), x_range=(0, 10), y_range=(0, 10))
    x, y = calib.to_data(50, 50)
    assert x == pytest.approx(5)
    assert y == pytest.approx(5)


def test_log_x_scale_maps_geometrically():
    calib = PixelCalibration(
        pixel_bbox=(0, 0, 100, 100), x_range=(1, 100), y_range=(0, 1), x_scale=ScaleType.LOG
    )
    x, _ = calib.to_data(50, 0)  # halfway across -> geometric mean
    assert x == pytest.approx(10, rel=1e-6)


def test_log_x_scale_rejects_non_positive_range():
    calib = PixelCalibration(
        pixel_bbox=(0, 0, 100, 100), x_range=(-1, 100), y_range=(0, 1), x_scale=ScaleType.LOG
    )
    with pytest.raises(ValueError, match="positive"):
        calib.to_data(50, 50)


def test_log_y_scale_maps_geometrically():
    # pixel y=50 (halfway up from the bottom, since pixel-down/data-up
    # inverts) -> geometric mean of y_range, mirroring log_x's test.
    calib = PixelCalibration(
        pixel_bbox=(0, 0, 100, 100), x_range=(0, 1), y_range=(1, 100), y_scale=ScaleType.LOG
    )
    _, y = calib.to_data(0, 50)
    assert y == pytest.approx(10, rel=1e-6)


def test_log_y_scale_rejects_non_positive_range():
    calib = PixelCalibration(
        pixel_bbox=(0, 0, 100, 100), x_range=(0, 1), y_range=(-1, 100), y_scale=ScaleType.LOG
    )
    with pytest.raises(ValueError, match="positive"):
        calib.to_data(50, 50)


def test_log_x_and_log_y_scale_can_both_be_set_independently():
    calib = PixelCalibration(
        pixel_bbox=(0, 0, 100, 100),
        x_range=(1, 100),
        y_range=(1, 100),
        x_scale=ScaleType.LOG,
        y_scale=ScaleType.LOG,
    )
    x, y = calib.to_data(50, 50)
    assert x == pytest.approx(10, rel=1e-6)
    assert y == pytest.approx(10, rel=1e-6)


def test_y_scale_defaults_to_linear():
    calib = PixelCalibration(pixel_bbox=(0, 0, 100, 100), x_range=(0, 10), y_range=(0, 10))
    assert calib.y_scale is ScaleType.LINEAR


def test_degenerate_pixel_bbox_does_not_divide_by_zero():
    calib = PixelCalibration(pixel_bbox=(10, 10, 10, 10), x_range=(0, 10), y_range=(0, 10))
    x, y = calib.to_data(10, 10)
    assert x == pytest.approx(0)
    assert y == pytest.approx(10)
