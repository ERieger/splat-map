"""TemporalFrameSelector's navigation controls: Previous/Next, the
+/- BIG_JUMP big-jump buttons, and the type-a-frame-number spin box --
all in addition to the pre-existing slider. Runs under the offscreen Qt
platform, same as every other GUI test here.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from vine360.gui.main_window import TemporalFrameSelector


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _selector_with_frames(n: int) -> TemporalFrameSelector:
    sel = TemporalFrameSelector()
    sel.set_frames([(f"f{i}", float(i), i % 3) for i in range(n)])
    return sel


def test_empty_state_disables_spin_box(qapp):
    sel = TemporalFrameSelector()
    assert sel.current_frame_id() is None
    assert not sel.frame_spin.isEnabled()


def test_set_frames_wires_spin_box_range_and_total_label(qapp):
    sel = _selector_with_frames(25)
    assert sel.current_frame_id() == "f0"
    assert sel.frame_spin.minimum() == 1
    assert sel.frame_spin.maximum() == 25
    assert sel.total_label.text() == "/ 25"


def test_next_and_prev_buttons_step_by_one(qapp):
    sel = _selector_with_frames(25)
    sel.next_btn.click()
    assert sel.slider.value() == 1
    assert sel.frame_spin.value() == 2  # kept in sync, 1-indexed
    sel.prev_btn.click()
    assert sel.slider.value() == 0
    assert sel.frame_spin.value() == 1


def test_big_jump_buttons_step_by_big_jump_and_clamp(qapp):
    sel = _selector_with_frames(25)
    sel.big_forward_btn.click()
    assert sel.slider.value() == TemporalFrameSelector.BIG_JUMP

    sel.big_back_btn.click()
    assert sel.slider.value() == 0
    sel.big_back_btn.click()  # already at the start -- must clamp, not go negative
    assert sel.slider.value() == 0

    sel.slider.setValue(24)
    sel.big_forward_btn.click()  # already near the end -- must clamp, not overshoot
    assert sel.slider.value() == 24


def test_typing_a_frame_number_jumps_to_it(qapp):
    sel = _selector_with_frames(25)
    sel.frame_spin.setValue(20)
    assert sel.slider.value() == 19  # spin is 1-indexed, slider is 0-indexed
    assert sel.current_frame_id() == "f19"


def test_spin_box_clamps_out_of_range_input(qapp):
    """The user asked that typed input be constrained to only valid
    values -- QSpinBox's own min/max enforces this natively."""
    sel = _selector_with_frames(25)
    sel.frame_spin.setValue(999)
    assert sel.frame_spin.value() == 25
    sel.frame_spin.setValue(-5)
    assert sel.frame_spin.value() == 1


def test_slider_and_spin_box_stay_in_sync_via_jump_to_frame_id(qapp):
    sel = _selector_with_frames(25)
    assert sel.jump_to_frame_id("f10") is True
    assert sel.frame_spin.value() == 11
    assert sel.jump_to_frame_id("no-such-frame") is False
