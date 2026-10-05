"""The Tesseract adapter returns word boxes in the strip's own pixels."""

import shutil

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from real_chart_bench.adapter.tesseract_ocr import ocr_words

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="no tesseract")


def test_reads_tick_labels_with_boxes_in_strip_pixels():
    im = Image.new("L", (300, 30), 255)
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=14)
    d.text((20, 8), "300", fill=0, font=font)
    d.text((200, 8), "-0.5", fill=0, font=font)
    words = ocr_words(np.asarray(im))
    texts = {w[0]: w for w in words}
    assert "300" in texts and "-0.5" in texts
    assert 15 <= texts["300"][1] <= 30 and texts["-0.5"][1] >= 190


def test_empty_strip():
    assert ocr_words(np.zeros((2, 2), dtype=np.uint8)) == []
