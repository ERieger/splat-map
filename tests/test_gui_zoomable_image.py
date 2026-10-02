"""ZoomableImageView: the Masks review preview must shrink to the space it's
given (a fixed 560px pixmap made the panel taller than a fullscreen window)
and zoom/pan from the full-resolution image. Offscreen Qt, like
test_gui_stage_indicator.py."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

import numpy as np
from PySide6.QtWidgets import QApplication

from vine360.gui.main_window import AppState, MasksPanel, ZoomableImageView, _numpy_to_pixmap


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _big_image() -> np.ndarray:
    return np.zeros((2000, 3000, 3), dtype=np.uint8)


def test_fits_large_image_inside_viewport(qapp):
    view = ZoomableImageView("empty")
    view.resize(400, 300)
    view.show()
    view.set_pixmap(_numpy_to_pixmap(_big_image()))
    qapp.processEvents()

    viewport = view.scroll_area.viewport().size()
    assert view.is_fit()
    assert view.image_label.width() <= viewport.width()
    assert view.image_label.height() <= viewport.height()
    # The widget never demands the image's size from its layout.
    assert view.minimumSizeHint().height() < 300
    # Full resolution is kept for zooming.
    assert view.pixmap().width() == 3000


def test_refits_on_resize_and_zooms(qapp):
    view = ZoomableImageView()
    view.resize(400, 300)
    view.show()
    view.set_pixmap(_numpy_to_pixmap(_big_image()))
    qapp.processEvents()
    small_fit = view.current_zoom()

    view.resize(800, 600)
    qapp.processEvents()
    assert view.current_zoom() > small_fit

    view.set_zoom(1.0)
    assert not view.is_fit()
    assert view.image_label.width() == 3000
    view.zoom_by(1.25)
    assert view.current_zoom() == pytest.approx(1.25)
    view.set_zoom(100.0)
    assert view.current_zoom() == ZoomableImageView.MAX_ZOOM

    # A new image keeps the chosen zoom (scrubbing frames stays zoomed).
    view.set_pixmap(_numpy_to_pixmap(np.zeros((100, 200), dtype=np.uint8)))
    assert view.current_zoom() == ZoomableImageView.MAX_ZOOM

    view.fit()
    assert view.is_fit()


def test_text_disables_zoom_controls(qapp):
    view = ZoomableImageView()
    view.set_pixmap(_numpy_to_pixmap(_big_image()))
    assert view.zoom_in_btn.isEnabled()
    view.set_text("Can't show overlay")
    assert not view.has_image()
    assert not view.zoom_in_btn.isEnabled()
    assert view.image_label.text() == "Can't show overlay"


def test_masks_panel_preview_no_longer_forces_tall_layout(qapp):
    panel = MasksPanel(AppState())
    panel.controls.setVisible(True)
    panel.preview_view.set_pixmap(_numpy_to_pixmap(_big_image()))
    # Previously >= 560px for the preview alone; now the preview contributes
    # only its small minimum, so the panel fits a normal window.
    assert panel.preview_view.minimumSizeHint().height() < 300
