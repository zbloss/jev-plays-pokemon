from PIL import Image

from jev_plays_pokemon.dialog_ocr import decode_dialog_text_ocr


class _FakeOCRResult:
    def __init__(self, txts: tuple[str, ...] | None) -> None:
        self.txts = txts


class _FakeOCREngine:
    """Stands in for `rapidocr.RapidOCR` - no real model load, per this ticket."""

    def __init__(self, txts: tuple[str, ...] | None) -> None:
        self._txts = txts
        self.last_img_content = None

    def __call__(self, img_content):
        self.last_img_content = img_content
        return _FakeOCRResult(self._txts)


def test_decode_dialog_text_ocr_returns_the_fake_engines_single_line():
    engine = _FakeOCREngine(("Hello there!",))
    screen = Image.new("RGB", (160, 144), color="white")

    text = decode_dialog_text_ocr(engine, screen)

    assert text == "Hello there!"


def test_decode_dialog_text_ocr_joins_multiple_detected_lines_with_newlines():
    engine = _FakeOCREngine(("Hello there!", "Welcome to the"))
    screen = Image.new("RGB", (160, 144), color="white")

    text = decode_dialog_text_ocr(engine, screen)

    assert text == "Hello there!\nWelcome to the"


def test_decode_dialog_text_ocr_strips_surrounding_whitespace():
    engine = _FakeOCREngine(("  Hello there!  ",))
    screen = Image.new("RGB", (160, 144), color="white")

    text = decode_dialog_text_ocr(engine, screen)

    assert text == "Hello there!"


def test_decode_dialog_text_ocr_returns_empty_string_when_nothing_detected():
    engine = _FakeOCREngine(None)
    screen = Image.new("RGB", (160, 144), color="white")

    text = decode_dialog_text_ocr(engine, screen)

    assert text == ""


def test_decode_dialog_text_ocr_passes_the_screen_image_straight_through():
    engine = _FakeOCREngine(("Hello there!",))
    screen = Image.new("RGB", (160, 144), color="white")

    decode_dialog_text_ocr(engine, screen)

    assert engine.last_img_content is screen
