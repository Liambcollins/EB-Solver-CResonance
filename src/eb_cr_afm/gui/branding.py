"""
Branding, theming, and header widgets for the CR2Beam GUI.

Centralises the application name, colour palette, Qt stylesheet, and the
ORNL-branded title bar so the rest of the GUI stays presentation-agnostic.

Notes
-----
The header shows a compact institutional mark.  If a real logo file is placed
at ``<repo>/assets/ornl_logo.(png|svg)`` (or ``ORNL_LOGO`` env var points at
one) it is used directly; otherwise a clean vector wordmark is drawn so the
app always looks finished.  The drawn mark is an original placeholder, not the
official ORNL trademark — drop in the approved asset for publication use.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QRectF, QSize
from PySide6.QtGui import (
    QColor, QPainter, QPen, QBrush, QFont, QPixmap, QLinearGradient,
    QPainterPath, QIcon,
)
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
APP_NAME = "CR2Beam"
APP_TAGLINE = "Two-Segment Euler–Bernoulli Contact-Resonance AFM Explorer"
APP_VERSION = "1.0"
ORG_NAME = "Oak Ridge National Laboratory"
ORG_SHORT = "ORNL"


# ---------------------------------------------------------------------------
# Palette  (light theme, exact ORNL brand colours from the official template)
#   dk1 373A36 · dk2/green 00662C · lime 7DBA00 · lt2 DBDCDB
#   accents: teal 00454D · blue 006BA6 · green 00B38F · orange FF9E1B · magenta B50093
# ---------------------------------------------------------------------------
class Palette:
    GREEN       = "#00662C"   # ORNL green (primary accent)
    GREEN_DARK  = "#004D21"   # pressed / hover-dark
    GREEN_LIGHT = "#00834A"   # hover
    LIME        = "#7DBA00"   # ORNL lime (secondary accent)
    INK         = "#373A36"   # ORNL graphite (primary text)
    INK_SOFT    = "#6B6F6A"   # secondary text
    BG          = "#eef0f1"   # window background
    CARD        = "#ffffff"   # group / card background
    BORDER      = "#d2d6d3"   # hairline borders
    FIELD       = "#ffffff"   # input background
    FIELD_DIS   = "#eceeed"   # disabled input
    HEADER_BG   = "#ffffff"   # title-bar background

    # Curve palette for plots — ORNL accent colours, readable on white
    C_BLUE   = "#006BA6"   # accent2
    C_RED    = "#FE5000"   # ORNL red-orange
    C_GREEN  = "#00662C"
    C_ORANGE = "#FF9E1B"   # accent5
    C_PURPLE = "#B50093"   # accent6 (magenta)
    C_CURSOR = "#00662C"


# ---------------------------------------------------------------------------
# Asset discovery
# ---------------------------------------------------------------------------
def _repo_root() -> Path:
    # branding.py -> gui -> eb_cr_afm -> src -> <repo root>
    return Path(__file__).resolve().parents[3]


def find_logo() -> Optional[Path]:
    """Return a user-supplied ORNL logo path if one exists, else None."""
    env = os.environ.get("ORNL_LOGO")
    candidates = []
    if env:
        candidates.append(Path(env))
    assets = _repo_root() / "assets"
    candidates += [
        assets / "ornl_logo.png",
        assets / "ornl_logo.svg",
        assets / "logo.png",
    ]
    for c in candidates:
        try:
            if c.is_file():
                return c
        except OSError:
            continue
    return None


# ---------------------------------------------------------------------------
# Drawn vector mark (fallback when no logo file is present)
# ---------------------------------------------------------------------------
class OrnlMark(QWidget):
    """A compact square emblem: green tile + stylized atom + 'ORNL'.

    Used as the square window/taskbar icon and as the header fallback when no
    official wordmark file is available.  ``use_logo_file`` is accepted for API
    symmetry but the mark is always drawn (the wide wordmark is handled by the
    header, not squeezed into this square).
    """

    def __init__(self, size: int = 46, parent=None, use_logo_file: bool = False):
        super().__init__(parent)
        self._sz = size
        self.setFixedSize(size, size)

    def sizeHint(self) -> QSize:
        return QSize(self._sz, self._sz)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = self.rect()

        # Rounded green tile with a subtle vertical gradient
        tile = QRectF(1, 1, r.width() - 2, r.height() - 2)
        grad = QLinearGradient(tile.topLeft(), tile.bottomLeft())
        grad.setColorAt(0.0, QColor(Palette.GREEN_LIGHT))
        grad.setColorAt(1.0, QColor(Palette.GREEN))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(grad))
        radius = r.width() * 0.22
        p.drawRoundedRect(tile, radius, radius)

        # Stylized atom: one nucleus + two lime orbits (science motif)
        cx, cy = tile.center().x(), tile.center().y() - tile.height() * 0.14
        rx, ry = tile.width() * 0.34, tile.height() * 0.15
        pen = QPen(QColor(Palette.LIME))
        pen.setWidthF(max(1.4, r.width() * 0.035))
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.save()
        p.translate(cx, cy)
        for angle in (35, -35):
            p.save()
            p.rotate(angle)
            p.drawEllipse(QRectF(-rx, -ry, 2 * rx, 2 * ry))
            p.restore()
        p.restore()
        p.setBrush(QBrush(QColor("#ffffff")))
        p.setPen(Qt.PenStyle.NoPen)
        nucleus = max(2.0, r.width() * 0.06)
        p.drawEllipse(QRectF(cx - nucleus, cy - nucleus, 2 * nucleus, 2 * nucleus))

        # 'ORNL' wordmark along the bottom
        f = QFont()
        f.setBold(True)
        f.setPixelSize(int(r.height() * 0.22))
        f.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 106)
        p.setFont(f)
        p.setPen(QPen(QColor("#ffffff")))
        band = QRectF(tile.left(), tile.bottom() - tile.height() * 0.34,
                      tile.width(), tile.height() * 0.30)
        p.drawText(band, Qt.AlignmentFlag.AlignCenter, "ORNL")
        p.end()


def app_icon() -> QIcon:
    """Square window / taskbar icon from the drawn mark.

    The official ORNL wordmark is wide (≈4:1) and looks squashed at icon
    sizes, so the icon always uses the compact square emblem (which is
    coloured to match the brand); the wordmark itself is shown in the header.
    """
    icon = QIcon()
    for s in (32, 64, 128, 256):
        w = OrnlMark(size=s, use_logo_file=False)
        w.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        pm = w.grab()
        icon.addPixmap(pm)
    return icon


# ---------------------------------------------------------------------------
# Header bar
# ---------------------------------------------------------------------------
class HeaderBar(QWidget):
    """Title strip: official ORNL wordmark + app name/tagline.

    When ``assets/ornl_logo.*`` (or ``$ORNL_LOGO``) resolves, the real ORNL
    wordmark is shown on the left; otherwise the drawn square emblem plus the
    lab name are used as a self-contained fallback.
    """

    _LOGO_H = 42  # rendered wordmark height (px)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("HeaderBar")
        self.setFixedHeight(64)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(18, 8, 20, 8)
        lay.setSpacing(14)

        logo_pm = self._load_logo()
        if logo_pm is not None:
            logo = QLabel()
            logo.setPixmap(logo_pm)
            logo.setFixedSize(logo_pm.width(), logo_pm.height())
            lay.addWidget(logo, 0, Qt.AlignmentFlag.AlignVCenter)

            # Thin brand-green divider between wordmark and app name
            rule = QLabel()
            rule.setObjectName("HeaderRule")
            rule.setFixedWidth(1)
            rule.setFixedHeight(self._LOGO_H - 4)
            lay.addWidget(rule, 0, Qt.AlignmentFlag.AlignVCenter)
        else:
            lay.addWidget(OrnlMark(size=46), 0, Qt.AlignmentFlag.AlignVCenter)

        # App name + tagline
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        name = QLabel(APP_NAME)
        name.setObjectName("AppName")
        tag = QLabel(APP_TAGLINE)
        tag.setObjectName("AppTagline")
        title_box.addWidget(name)
        title_box.addWidget(tag)
        lay.addLayout(title_box)

        lay.addStretch(1)

        # Right side: application context (or lab name if no wordmark)
        org_box = QVBoxLayout()
        org_box.setSpacing(0)
        if logo_pm is None:
            org = QLabel(ORG_NAME)
            org.setObjectName("OrgName")
            org.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            org_box.addWidget(org)
        sub = QLabel("Contact-Resonance / PFM modelling")
        sub.setObjectName("OrgSub")
        sub.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        org_box.addWidget(sub)
        lay.addLayout(org_box)

    def _load_logo(self) -> Optional[QPixmap]:
        path = find_logo()
        if path is None:
            return None
        pm = QPixmap(str(path))
        if pm.isNull():
            return None
        dpr = float(self.devicePixelRatioF()) if hasattr(self, "devicePixelRatioF") else 1.0
        target_h = int(self._LOGO_H * max(1.0, dpr))
        pm = pm.scaledToHeight(target_h, Qt.TransformationMode.SmoothTransformation)
        pm.setDevicePixelRatio(max(1.0, dpr))
        return pm

    def paintEvent(self, event):
        # White background with a green->lime accent rule along the bottom.
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor(Palette.HEADER_BG))
        grad = QLinearGradient(0, 0, self.width(), 0)
        grad.setColorAt(0.0, QColor(Palette.GREEN))
        grad.setColorAt(1.0, QColor(Palette.LIME))
        h = 3
        p.fillRect(0, self.height() - h, self.width(), h, QBrush(grad))
        p.end()


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------
def build_stylesheet() -> str:
    P = Palette
    return f"""
* {{
    font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif;
    font-size: 10pt;
    color: {P.INK};
}}
QMainWindow, QWidget#Central {{ background: {P.BG}; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}

/* Header */
QLabel#AppName    {{ color: {P.GREEN}; font-size: 19pt; font-weight: 800; }}
QLabel#AppTagline {{ color: {P.INK_SOFT}; font-size: 9pt; }}
QLabel#OrgName    {{ color: {P.INK}; font-size: 10.5pt; font-weight: 700; }}
QLabel#OrgSub     {{ color: {P.INK_SOFT}; font-size: 8.5pt; }}
QLabel#HeaderRule {{ background: {P.BORDER}; }}

/* Group boxes as cards */
QGroupBox {{
    background: {P.CARD};
    border: 1px solid {P.BORDER};
    border-radius: 8px;
    margin-top: 14px;
    padding: 10px 10px 8px 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 1px 8px;
    color: {P.GREEN};
    background: {P.CARD};
    border: 1px solid {P.BORDER};
    border-radius: 5px;
    font-size: 9.5pt;
    font-weight: 700;
}}

QLabel {{ background: transparent; }}
QLabel#SubHead {{ color: {P.INK_SOFT}; font-weight: 700; font-size: 8.5pt; }}

/* Inputs */
QDoubleSpinBox, QSpinBox, QComboBox {{
    background: {P.FIELD};
    border: 1px solid {P.BORDER};
    border-radius: 5px;
    padding: 3px 6px;
    min-height: 20px;
    selection-background-color: {P.GREEN_LIGHT};
    selection-color: white;
}}
QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{ border: 1px solid {P.GREEN_LIGHT}; }}
QDoubleSpinBox:disabled, QSpinBox:disabled, QComboBox:disabled {{
    background: {P.FIELD_DIS}; color: {P.INK_SOFT};
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {P.CARD};
    border: 1px solid {P.BORDER};
    selection-background-color: {P.GREEN_LIGHT};
    selection-color: white;
    outline: none;
}}

/* Buttons */
QPushButton {{
    background: {P.CARD};
    border: 1px solid {P.BORDER};
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 600;
    color: {P.INK};
}}
QPushButton:hover  {{ border-color: {P.GREEN_LIGHT}; color: {P.GREEN_DARK}; }}
QPushButton:pressed {{ background: {P.FIELD_DIS}; }}
QPushButton[primary="true"] {{
    background: {P.GREEN}; color: white; border: 1px solid {P.GREEN};
    padding: 8px 12px;
}}
QPushButton[primary="true"]:hover   {{ background: {P.GREEN_LIGHT}; border-color: {P.GREEN_LIGHT}; }}
QPushButton[primary="true"]:pressed {{ background: {P.GREEN_DARK}; }}

/* Checkboxes */
QCheckBox {{ spacing: 7px; background: transparent; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {P.BORDER}; border-radius: 4px; background: {P.FIELD};
}}
QCheckBox::indicator:hover  {{ border-color: {P.GREEN_LIGHT}; }}
QCheckBox::indicator:checked {{
    background: {P.GREEN}; border-color: {P.GREEN};
    image: url();
}}

/* Sliders */
QSlider::groove:horizontal {{
    height: 5px; background: {P.BORDER}; border-radius: 3px;
}}
QSlider::sub-page:horizontal {{ background: {P.LIME}; border-radius: 3px; }}
QSlider::handle:horizontal {{
    background: {P.GREEN}; border: 2px solid white;
    width: 15px; height: 15px; margin: -6px 0; border-radius: 9px;
}}
QSlider::handle:horizontal:hover {{ background: {P.GREEN_LIGHT}; }}

/* Tabs */
QTabWidget::pane {{
    border: 1px solid {P.BORDER}; border-radius: 8px;
    background: {P.CARD}; top: -1px;
}}
QTabBar::tab {{
    background: transparent; color: {P.INK_SOFT};
    padding: 7px 18px; margin-right: 2px;
    border: none; border-bottom: 3px solid transparent;
    font-weight: 600;
}}
QTabBar::tab:selected {{ color: {P.GREEN}; border-bottom: 3px solid {P.GREEN}; }}
QTabBar::tab:hover:!selected {{ color: {P.INK}; }}

/* Menu bar */
QMenuBar {{ background: {P.CARD}; border-bottom: 1px solid {P.BORDER}; }}
QMenuBar::item {{ background: transparent; padding: 5px 10px; }}
QMenuBar::item:selected {{ background: {P.BG}; border-radius: 4px; }}
QMenu {{ background: {P.CARD}; border: 1px solid {P.BORDER}; }}
QMenu::item:selected {{ background: {P.GREEN_LIGHT}; color: white; }}

/* Status bar */
QStatusBar {{ background: {P.CARD}; border-top: 1px solid {P.BORDER}; color: {P.INK_SOFT}; }}
QStatusBar QLabel {{ color: {P.INK_SOFT}; }}
QStatusBar QLabel#ResLabel {{ color: {P.GREEN}; font-weight: 700; }}

/* Progress bar */
QProgressBar {{
    border: 1px solid {P.BORDER}; border-radius: 5px;
    background: {P.FIELD}; height: 14px; text-align: center;
    color: {P.INK}; font-size: 8pt;
}}
QProgressBar::chunk {{ background: {P.LIME}; border-radius: 4px; }}

/* Splitter */
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:horizontal {{ width: 6px; }}
QSplitter::handle:vertical {{ height: 6px; }}

/* Scrollbars */
QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {P.BORDER}; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {P.INK_SOFT}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {P.BORDER}; border-radius: 5px; min-width: 24px; }}
QScrollBar::handle:horizontal:hover {{ background: {P.INK_SOFT}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
"""
