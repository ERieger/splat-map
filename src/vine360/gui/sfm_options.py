"""The Pose estimation panel's "Advanced options" editor (docs/adr/0040).

Built entirely from vine360.sfm.options.OPTION_SPECS -- one row per
SfmConfig field, grouped as the specs group them -- so a new option only
needs a field and a spec, never GUI code. Rows the current engine variant
can't use are hidden; rows that only matter alongside another choice (e.g.
the sequential overlap with the sequential matcher) are disabled until it's
chosen. Every tooltip is the spec's own help text.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from vine360.sfm.options import (
    GROUPS,
    OPTION_SPECS,
    PRESETS,
    VARIANT_PROJECTIONS,
    OptionSpec,
    SfmConfig,
    describe,
    preset_config,
)


class _PathField(QWidget):
    changed = Signal()

    def __init__(self, spec: OptionSpec):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("No file chosen")
        self.edit.textChanged.connect(self.changed)
        row.addWidget(self.edit, stretch=1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        self._title = spec.label

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, self._title, self.edit.text(), "Vocabulary trees (*.bin);;All files (*)")
        if path:
            self.edit.setText(path)

    def value(self) -> str:
        return self.edit.text().strip()

    def set_value(self, value: str) -> None:
        self.edit.setText(value)


def _make_field(spec: OptionSpec):
    """(widget, getter, setter, change signal) for one option."""
    if spec.kind == "bool":
        widget = QCheckBox()
        return widget, widget.isChecked, widget.setChecked, widget.toggled
    if spec.kind == "choice":
        widget = QComboBox()
        for value, label in spec.choices:
            widget.addItem(label, userData=value)

        def set_choice(value, widget=widget):
            index = widget.findData(value)
            widget.setCurrentIndex(index if index >= 0 else 0)

        return widget, widget.currentData, set_choice, widget.currentIndexChanged
    if spec.kind == "path":
        widget = _PathField(spec)
        return widget, widget.value, widget.set_value, widget.changed
    if spec.kind == "int":
        widget = QSpinBox()
        widget.setRange(int(spec.minimum), int(spec.maximum))
        if spec.step:
            widget.setSingleStep(int(spec.step))
    else:
        widget = QDoubleSpinBox()
        widget.setDecimals(spec.decimals)
        widget.setRange(spec.minimum, spec.maximum)
        if spec.step:
            widget.setSingleStep(spec.step)
    if spec.special_value_text:
        widget.setSpecialValueText(spec.special_value_text)
    widget.setKeyboardTracking(False)
    getter = widget.value
    if spec.kind == "float":
        # A spin box rounds to its decimals (the default peak threshold
        # 0.00667 shows as 0.0067) -- read the rounded default back as the
        # exact default, so an untouched field never counts as changed.
        default = getattr(SfmConfig(), spec.key)

        def getter(widget=widget, default=default):
            return default if widget.value() == round(default, widget.decimals()) else widget.value()

    return widget, getter, widget.setValue, widget.valueChanged


class SfmOptionsEditor(QWidget):
    """Collapsible "Advanced options" section: a toggle row (with a
    one-line summary of what's changed), then -- when expanded -- presets,
    Reset, and the grouped option rows in a scroll area."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._variant = VARIANT_PROJECTIONS
        self._loading = False
        self._fields: dict[str, tuple] = {}  # key -> (widget, getter, setter, label, form)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        toggle_row = QHBoxLayout()
        self.toggle_btn = QToolButton()
        self.toggle_btn.setText("Advanced options")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setArrowType(Qt.RightArrow)
        self.toggle_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_btn.toggled.connect(self._on_toggled)
        toggle_row.addWidget(self.toggle_btn)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("color: palette(mid);")
        toggle_row.addWidget(self.summary_label, stretch=1)
        outer.addLayout(toggle_row)

        self.body = QWidget()
        body_layout = QVBoxLayout(self.body)
        body_layout.setContentsMargins(0, 0, 0, 0)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Custom", userData=None)
        for name in PRESETS:
            self.preset_combo.addItem(name, userData=name)
        self.preset_combo.setToolTip(
            "Starting points. Fast preview: smaller images, fewer features and neighbours. Thorough: larger "
            "images, more features and neighbours, guided matching. Neither is tuned on real vineyard footage yet."
        )
        self.preset_combo.activated.connect(self._on_preset)
        preset_row.addWidget(self.preset_combo)
        reset_btn = QPushButton("Reset to defaults")
        reset_btn.clicked.connect(lambda: self.set_config(SfmConfig()))
        preset_row.addWidget(reset_btn)
        preset_row.addStretch()
        body_layout.addLayout(preset_row)

        self.problems_label = QLabel("")
        self.problems_label.setWordWrap(True)
        self.problems_label.setStyleSheet("color: #c0392b;")
        self.problems_label.setVisible(False)
        body_layout.addWidget(self.problems_label)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 4, 0)
        forms: dict[str, QFormLayout] = {}
        self._group_boxes: dict[str, QGroupBox] = {}
        for group in GROUPS:
            box = QGroupBox(group)
            forms[group] = QFormLayout(box)
            self._group_boxes[group] = box
            content_layout.addWidget(box)
        content_layout.addStretch()

        for spec in OPTION_SPECS:
            widget, getter, setter, signal = _make_field(spec)
            widget.setToolTip(spec.help)
            label = QLabel(spec.label + ":")
            label.setToolTip(spec.help)
            forms[spec.group].addRow(label, widget)
            signal.connect(self._on_field_changed)
            self._fields[spec.key] = (widget, getter, setter, label, forms[spec.group])

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setMinimumHeight(260)
        self.scroll.setMaximumHeight(640)
        self.scroll.setWidget(content)
        body_layout.addWidget(self.scroll)

        self.body.setVisible(False)
        outer.addWidget(self.body)
        self.set_config(SfmConfig())

    # -- public API ------------------------------------------------------

    def config(self) -> SfmConfig:
        """Everything as shown, including rows hidden for this variant (use
        SfmConfig.for_variant before running)."""
        return SfmConfig.from_dict({key: field[1]() for key, field in self._fields.items()})

    def set_config(self, config: SfmConfig) -> None:
        self._loading = True
        try:
            for key, field in self._fields.items():
                field[2](getattr(config, key))
        finally:
            self._loading = False
        self._on_field_changed()

    def set_variant(self, variant: str) -> None:
        self._variant = variant
        self._refresh_rows()
        self._refresh_summary()

    def problems(self) -> list[str]:
        return self.config().for_variant(self._variant).validate(self._variant)

    def set_expanded(self, expanded: bool) -> None:
        self.toggle_btn.setChecked(expanded)

    # -- internals -------------------------------------------------------

    def _on_toggled(self, expanded: bool) -> None:
        self.toggle_btn.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.body.setVisible(expanded)

    def _on_preset(self) -> None:
        name = self.preset_combo.currentData()
        if name is not None:
            self.set_config(preset_config(name))

    def _on_field_changed(self, *_args) -> None:
        if self._loading:
            return
        self._refresh_rows()
        self._refresh_summary()
        self._sync_preset_combo()
        self.changed.emit()

    def _refresh_rows(self) -> None:
        config = self.config()
        visible_per_group: dict[str, int] = {}
        for spec in OPTION_SPECS:
            widget, _get, _set, label, form = self._fields[spec.key]
            applies = self._variant in spec.variants
            form.setRowVisible(widget, applies)
            widget.setEnabled(config.is_enabled(spec.key))
            label.setEnabled(config.is_enabled(spec.key))
            visible_per_group[spec.group] = visible_per_group.get(spec.group, 0) + int(applies)
        for group, box in self._group_boxes.items():
            box.setVisible(visible_per_group.get(group, 0) > 0)

    def _refresh_summary(self) -> None:
        effective = self.config().for_variant(self._variant)
        changed = describe(effective.replace(camera_model=SfmConfig().camera_model)) if self._variant != VARIANT_PROJECTIONS else describe(effective)
        self.summary_label.setText(f"Changed: {changed}" if changed else "All defaults")
        self.summary_label.setToolTip(changed)
        problems = effective.validate(self._variant)
        self.problems_label.setText("\n".join(f"• {p}" for p in problems))
        self.problems_label.setVisible(bool(problems))

    def _sync_preset_combo(self) -> None:
        current = self.config().to_dict()
        match = next((name for name in PRESETS if preset_config(name).to_dict() == current), None)
        index = self.preset_combo.findData(match)
        self.preset_combo.setCurrentIndex(index if index >= 0 else 0)
