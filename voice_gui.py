import json
import os
import platform
import sys
import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard
from PySide6 import QtCore, QtGui, QtWidgets


APP_NAME = "Voice"


def _app_config_path() -> str:
    system = platform.system().lower()
    if system == "windows":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, APP_NAME, "config.json")
    if system == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
        return os.path.join(base, APP_NAME, "config.json")
    base = os.path.expanduser("~/.config")
    return os.path.join(base, APP_NAME.lower(), "config.json")


def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _key_to_string(k) -> str:
    if isinstance(k, keyboard.KeyCode) and k.char:
        return k.char.lower()
    if isinstance(k, keyboard.Key):
        return k.name.lower()
    return "f9"


def _string_to_key(s: str):
    s = (s or "").strip().lower()
    if not s:
        return keyboard.Key.f9
    if len(s) == 1:
        return keyboard.KeyCode.from_char(s)
    try:
        return getattr(keyboard.Key, s)
    except AttributeError:
        # Fallback to f9 instead of crashing on bad config.
        return keyboard.Key.f9


@dataclass
class AppConfig:
    hotkey: str = "f9"
    model: str = "base"  # multilingual -> EN/FR
    language: str = "auto"  # auto|en|fr
    device: str = "cpu"  # cpu|cuda
    compute_type: str = "int8"  # int8 (cpu), float16 (cuda), etc.
    paste_after_copy: bool = False
    vad_filter: bool = False
    always_on_top: bool = True
    show_bar: bool = True
    show_transcript_toast: bool = True
    window_pos: Optional[Tuple[int, int]] = None


def load_config() -> AppConfig:
    path = _app_config_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cfg = AppConfig(**{k: raw[k] for k in raw.keys() if k in AppConfig.__annotations__})
        return cfg
    except FileNotFoundError:
        return AppConfig()
    except Exception:
        # Bad config shouldn't brick the app.
        return AppConfig()


def save_config(cfg: AppConfig) -> None:
    path = _app_config_path()
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


class Recorder:
    def __init__(self, samplerate: int = 16000):
        self._samplerate = samplerate
        self._frames = []
        self._lock = threading.Lock()
        self._stream: Optional[sd.InputStream] = None
        self._recording = False

    @property
    def recording(self) -> bool:
        with self._lock:
            return self._recording

    def start(self) -> None:
        with self._lock:
            if self._recording:
                return
            self._frames = []
            self._recording = True

        def _callback(indata, frames, time_info, status):
            # Keep callback minimal.
            with self._lock:
                if self._recording:
                    self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self._samplerate,
            channels=1,
            dtype="float32",
            callback=_callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        with self._lock:
            if not self._recording:
                return np.zeros((0,), dtype=np.float32)
            self._recording = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            if not self._frames:
                return np.zeros((0,), dtype=np.float32)
            audio = np.concatenate(self._frames, axis=0).reshape(-1).astype(np.float32)
            self._frames = []
            return audio


class TranscribeWorker(QtCore.QObject):
    finished = QtCore.Signal(str, str, float)  # text, language, prob
    failed = QtCore.Signal(str)

    def __init__(self, model: WhisperModel, audio: np.ndarray, cfg: AppConfig):
        super().__init__()
        self._model = model
        self._audio = audio
        self._cfg = cfg

    @QtCore.Slot()
    def run(self):
        try:
            language = None if self._cfg.language.strip().lower() == "auto" else self._cfg.language.strip().lower()
            segments, info = self._model.transcribe(
                self._audio,
                language=language,
                beam_size=5,
                vad_filter=bool(self._cfg.vad_filter),
            )
            text = "".join(seg.text for seg in segments).strip()
            detected_lang = getattr(info, "language", "") or ""
            prob = float(getattr(info, "language_probability", 0.0) or 0.0)
            self.finished.emit(text, detected_lang, prob)
        except Exception as e:
            self.failed.emit(str(e))


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, cfg: AppConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} Settings")
        self.setModal(True)
        self._cfg = cfg

        self.hotkey_edit = QtWidgets.QLineEdit(cfg.hotkey)
        self.hotkey_edit.setPlaceholderText("Press a key (e.g. f9, space, a)")
        self.hotkey_edit.installEventFilter(self)

        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "base.en", "small.en"])
        idx = self.model_combo.findText(cfg.model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

        self.lang_combo = QtWidgets.QComboBox()
        self.lang_combo.addItem("Auto", "auto")
        self.lang_combo.addItem("English", "en")
        self.lang_combo.addItem("French", "fr")
        idx = self.lang_combo.findData(cfg.language)
        if idx >= 0:
            self.lang_combo.setCurrentIndex(idx)

        self.paste_chk = QtWidgets.QCheckBox("Auto-paste after copying (Ctrl/Cmd+V)")
        self.paste_chk.setChecked(bool(cfg.paste_after_copy))

        self.vad_chk = QtWidgets.QCheckBox("Enable VAD filter (helps with long silences)")
        self.vad_chk.setChecked(bool(cfg.vad_filter))

        self.on_top_chk = QtWidgets.QCheckBox("Always on top")
        self.on_top_chk.setChecked(bool(cfg.always_on_top))

        self.show_bar_chk = QtWidgets.QCheckBox("Show Flow bar")
        self.show_bar_chk.setChecked(bool(cfg.show_bar))

        self.toast_chk = QtWidgets.QCheckBox("Show transcript bubble after release")
        self.toast_chk.setChecked(bool(cfg.show_transcript_toast))

        form = QtWidgets.QFormLayout()
        form.addRow("Hotkey (push-to-talk)", self.hotkey_edit)
        form.addRow("Model", self.model_combo)
        form.addRow("Language", self.lang_combo)
        form.addRow("", self.paste_chk)
        form.addRow("", self.vad_chk)
        form.addRow("", self.on_top_chk)
        form.addRow("", self.show_bar_chk)
        form.addRow("", self.toast_chk)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(btns)
        self.setLayout(layout)

    def eventFilter(self, obj, event):
        if obj is self.hotkey_edit and event.type() == QtCore.QEvent.KeyPress:
            key = event.key()
            text = QtGui.QKeySequence(key).toString().lower()
            # QKeySequence returns "F9" etc. Normalize some common ones.
            text = text.replace(" ", "").lower()
            if text:
                self.hotkey_edit.setText(text)
                return True
        return super().eventFilter(obj, event)

    def updated_config(self) -> AppConfig:
        cfg = AppConfig(**asdict(self._cfg))
        cfg.hotkey = (self.hotkey_edit.text() or "f9").strip().lower()
        cfg.model = self.model_combo.currentText().strip()
        cfg.language = str(self.lang_combo.currentData())
        cfg.paste_after_copy = bool(self.paste_chk.isChecked())
        cfg.vad_filter = bool(self.vad_chk.isChecked())
        cfg.always_on_top = bool(self.on_top_chk.isChecked())
        cfg.show_bar = bool(self.show_bar_chk.isChecked())
        cfg.show_transcript_toast = bool(self.toast_chk.isChecked())
        return cfg


class IndicatorWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = "idle"  # idle|rec|work
        self._phase = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._tick)

        self.setFixedHeight(18)
        self.setFixedWidth(118)

    def set_mode(self, mode: str):
        mode = (mode or "idle").strip().lower()
        if mode not in ("idle", "rec", "work"):
            mode = "idle"
        if mode == self._mode:
            return
        self._mode = mode
        self._phase = 0.0
        if mode in ("rec", "work"):
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def _tick(self):
        self._phase += 0.35
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        r = self.rect()

        if self._mode == "idle":
            n = 11
            dot = 3
            gap = 7
            total = n * dot + (n - 1) * gap
            x0 = int((r.width() - total) / 2)
            y0 = int((r.height() - dot) / 2)
            for i in range(n):
                c = QtGui.QColor(255, 255, 255, 80)
                if i in (4, 5, 6):
                    c = QtGui.QColor(255, 255, 255, 110)
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(c)
                p.drawEllipse(QtCore.QRectF(x0 + i * (dot + gap), y0, dot, dot))
            return

        if self._mode == "work":
            # 3-dot bounce
            n = 3
            dot = 5
            gap = 9
            total = n * dot + (n - 1) * gap
            x0 = int((r.width() - total) / 2)
            y = int((r.height() - dot) / 2)
            active = int(self._phase) % n
            for i in range(n):
                a = 90 if i != active else 180
                c = QtGui.QColor(255, 255, 255, a)
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(c)
                p.drawEllipse(QtCore.QRectF(x0 + i * (dot + gap), y, dot, dot))
            return

        # recording: equalizer bars
        n = 9
        bar_w = 4
        gap = 4
        max_h = 16
        min_h = 6
        total = n * bar_w + (n - 1) * gap
        x0 = int((r.width() - total) / 2)
        base_y = int(r.height() / 2)

        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(255, 255, 255, 220))
        for i in range(n):
            t = self._phase + i * 0.55
            # a few combined sines to avoid looking too uniform
            v = (0.55 + 0.45 * np.sin(t)) * (0.55 + 0.45 * np.sin(t * 0.6 + 1.2))
            h = min_h + (max_h - min_h) * float(v)
            x = x0 + i * (bar_w + gap)
            rect = QtCore.QRectF(x, base_y - h / 2, bar_w, h)
            p.drawRoundedRect(rect, 2, 2)


class StopButton(QtWidgets.QAbstractButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFixedSize(22, 22)
        self.setToolTip("Stop")

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        r = self.rect().adjusted(1, 1, -1, -1)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor("#EF4444"))
        p.drawEllipse(r)
        # white stop square
        s = 8
        cx = r.center().x()
        cy = r.center().y()
        p.setBrush(QtGui.QColor("#FFFFFF"))
        p.drawRoundedRect(QtCore.QRectF(cx - s / 2, cy - s / 2, s, s), 2, 2)


class Overlay(QtWidgets.QWidget):
    open_settings = QtCore.Signal()
    quit_requested = QtCore.Signal()
    stop_requested = QtCore.Signal()

    def __init__(self, cfg: AppConfig):
        flags = QtCore.Qt.FramelessWindowHint | QtCore.Qt.Tool
        if cfg.always_on_top:
            flags |= QtCore.Qt.WindowStaysOnTopHint
        super().__init__(None, flags)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setWindowTitle(APP_NAME)
        self.setFocusPolicy(QtCore.Qt.NoFocus)

        self._cfg = cfg
        self._drag_pos: Optional[QtCore.QPoint] = None

        self.container = QtWidgets.QFrame()
        self.container.setObjectName("container")

        self.indicator = IndicatorWidget()
        self.stop_btn = StopButton()
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        self.stop_btn.hide()

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(14, 10, 14, 10)
        row.setSpacing(10)
        row.addWidget(self.indicator, 0, QtCore.Qt.AlignVCenter)
        row.addWidget(self.stop_btn, 0, QtCore.Qt.AlignVCenter)

        self.container.setLayout(row)

        root = QtWidgets.QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.container)
        self.setLayout(root)

        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(32)
        shadow.setOffset(0, 12)
        shadow.setColor(QtGui.QColor(0, 0, 0, 120))
        self.container.setGraphicsEffect(shadow)

        self._apply_style()
        self._autosize()
        self.set_state_idle()

    def _autosize(self):
        self.container.adjustSize()
        self.adjustSize()

    def _apply_style(self):
        # Wispr-ish: near-black pill with subtle border + shadow.
        bg = "rgba(10, 12, 16, 235)"
        border = "rgba(255, 255, 255, 0.12)"
        self.setStyleSheet(
            f"""
            #container {{
              background: {bg};
              border: 1px solid {border};
              border-radius: 999px;
            }}
            """
        )

    def set_config(self, cfg: AppConfig):
        self._cfg = cfg
        flags = QtCore.Qt.FramelessWindowHint | QtCore.Qt.Tool
        if cfg.always_on_top:
            flags |= QtCore.Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.show() if cfg.show_bar else self.hide()
        self._apply_style()
        self.set_state_idle()

    def set_state_idle(self):
        self.indicator.set_mode("idle")
        self.stop_btn.hide()
        self._autosize()

    def set_state_recording(self):
        self.indicator.set_mode("rec")
        self.stop_btn.show()
        self._autosize()

    def set_state_transcribing(self):
        self.indicator.set_mode("work")
        self.stop_btn.hide()
        self._autosize()

    def show_transcript(self, text: str, lang: str = "", prob: float = 0.0):
        self.set_state_idle()
        if not self._cfg.show_transcript_toast:
            QtWidgets.QToolTip.showText(self.mapToGlobal(QtCore.QPoint(self.width() // 2, 0)), "Copied", self)
            return
        prefix = f"{lang.upper()} " if lang else ""
        msg = f"{prefix}{text}".strip()
        QtWidgets.QToolTip.showText(self.mapToGlobal(QtCore.QPoint(self.width() // 2, 0)), msg, self)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton:
            self.open_settings.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            return
        if event.button() == QtCore.Qt.RightButton:
            self._show_context_menu(event.globalPosition().toPoint())
            event.accept()
            return

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self._drag_pos is not None and event.buttons() & QtCore.Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_pos = None
            event.accept()

    def _show_context_menu(self, global_pos: QtCore.QPoint):
        menu = QtWidgets.QMenu(self)
        act_settings = menu.addAction("Settings...")
        act_settings.triggered.connect(self.open_settings.emit)
        menu.addSeparator()
        act_quit = menu.addAction("Quit")
        act_quit.triggered.connect(self.quit_requested.emit)
        menu.exec(global_pos)


class HotkeyListener(QtCore.QObject):
    pressed = QtCore.Signal()
    released = QtCore.Signal()

    def __init__(self, hotkey: str):
        super().__init__()
        self._hotkey_str = hotkey
        self._hotkey = _string_to_key(hotkey)
        self._listener: Optional[keyboard.Listener] = None

    def set_hotkey(self, hotkey: str):
        self._hotkey_str = hotkey
        self._hotkey = _string_to_key(hotkey)

    def start(self):
        def on_press(k):
            if k == self._hotkey:
                self.pressed.emit()

        def on_release(k):
            if k == self._hotkey:
                self.released.emit()

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.daemon = True
        self._listener.start()

    def stop(self):
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None


class App(QtCore.QObject):
    def __init__(self):
        super().__init__()
        # Ensure HF telemetry is off; this app should never ship content off-device.
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

        self.cfg = load_config()
        self.overlay = Overlay(self.cfg)
        self.overlay.open_settings.connect(self.show_settings)
        self.overlay.quit_requested.connect(self.quit)
        self.overlay.stop_requested.connect(self.on_stop_requested)

        self.recorder = Recorder(samplerate=16000)
        self._model_lock = threading.Lock()
        self._model: Optional[WhisperModel] = None
        self._transcribing = False

        self.hotkeys = HotkeyListener(self.cfg.hotkey)
        self.hotkeys.pressed.connect(self.on_ptt_pressed)
        self.hotkeys.released.connect(self.on_ptt_released)
        self.hotkeys.start()

        self.tray = self._create_tray()

        if self.cfg.window_pos:
            self.overlay.move(self.cfg.window_pos[0], self.cfg.window_pos[1])
        else:
            self._place_default()

        self.overlay.show() if self.cfg.show_bar else self.overlay.hide()

    def _place_default(self):
        screen = QtGui.QGuiApplication.primaryScreen()
        if not screen:
            return
        geo = screen.availableGeometry()
        x = geo.x() + (geo.width() - self.overlay.width()) // 2
        y = geo.y() + geo.height() - self.overlay.height() - 60
        self.overlay.move(x, y)

    def _create_tray(self) -> QtWidgets.QSystemTrayIcon:
        style = QtWidgets.QApplication.style()
        icon = style.standardIcon(QtWidgets.QStyle.SP_MediaVolume)
        tray = QtWidgets.QSystemTrayIcon(icon)
        tray.setToolTip(APP_NAME)
        menu = QtWidgets.QMenu()

        self.act_show_bar = menu.addAction("Show Flow bar")
        self.act_show_bar.setCheckable(True)
        self.act_show_bar.setChecked(bool(self.cfg.show_bar))
        self.act_show_bar.triggered.connect(self.toggle_bar)

        act_settings = menu.addAction("Settings...")
        act_settings.triggered.connect(self.show_settings)
        act_quit = menu.addAction("Quit")
        act_quit.triggered.connect(self.quit)
        tray.setContextMenu(menu)
        tray.show()
        return tray

    def _get_model(self) -> WhisperModel:
        with self._model_lock:
            if self._model is None:
                self._model = WhisperModel(self.cfg.model, device=self.cfg.device, compute_type=self.cfg.compute_type)
            return self._model

    @QtCore.Slot()
    def on_ptt_pressed(self):
        if self._transcribing:
            return
        if self.recorder.recording:
            return
        try:
            self.recorder.start()
            if self.cfg.show_bar:
                self.overlay.set_state_recording()
        except Exception as e:
            self.overlay.show_transcript(f"Mic error: {e}")

    @QtCore.Slot()
    def on_ptt_released(self):
        if not self.recorder.recording:
            return
        audio = self.recorder.stop()
        if audio.size == 0:
            if self.cfg.show_bar:
                self.overlay.set_state_idle()
            return
        if self._transcribing:
            if self.cfg.show_bar:
                self.overlay.set_state_idle()
            return

        self._transcribing = True
        if self.cfg.show_bar:
            self.overlay.set_state_transcribing()

        model = self._get_model()
        self._worker = TranscribeWorker(model, audio, self.cfg)
        self._thread = QtCore.QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self.on_transcribed)
        self._worker.failed.connect(self.on_transcribe_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    @QtCore.Slot(str, str, float)
    def on_transcribed(self, text: str, lang: str, prob: float):
        self._transcribing = False
        text = (text or "").strip()
        if not text:
            if self.cfg.show_bar:
                self.overlay.show_transcript("...")
            return

        QtWidgets.QApplication.clipboard().setText(text)
        if self.cfg.paste_after_copy:
            self._paste_now()

        if self.cfg.show_bar:
            self.overlay.show_transcript(text, lang=lang, prob=prob)

    @QtCore.Slot(str)
    def on_transcribe_failed(self, err: str):
        self._transcribing = False
        if self.cfg.show_bar:
            self.overlay.show_transcript(f"ASR error: {err}")

    def _paste_now(self):
        is_macos = platform.system().lower() == "darwin"
        mod = keyboard.Key.cmd if is_macos else keyboard.Key.ctrl
        ctl = keyboard.Controller()
        # tiny delay so the active app regains focus after key-up
        time.sleep(0.05)
        try:
            with ctl.pressed(mod):
                ctl.press("v")
                ctl.release("v")
        except Exception:
            pass

    @QtCore.Slot()
    def show_settings(self):
        dlg = SettingsDialog(self.cfg, parent=self.overlay)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        new_cfg = dlg.updated_config()

        # Persist window position.
        pos = self.overlay.pos()
        new_cfg.window_pos = (pos.x(), pos.y())

        self.cfg = new_cfg
        save_config(self.cfg)

        # Apply live.
        self.overlay.set_config(self.cfg)
        self.hotkeys.set_hotkey(self.cfg.hotkey)
        with self._model_lock:
            self._model = None  # reload with new model settings on next use
        if hasattr(self, "act_show_bar"):
            self.act_show_bar.setChecked(bool(self.cfg.show_bar))

    @QtCore.Slot()
    def toggle_bar(self):
        self.cfg.show_bar = bool(self.act_show_bar.isChecked())
        save_config(self.cfg)
        self.overlay.set_config(self.cfg)

    @QtCore.Slot()
    def on_stop_requested(self):
        # Allows stopping via UI while recording.
        if self.recorder.recording:
            self.on_ptt_released()

    @QtCore.Slot()
    def quit(self):
        pos = self.overlay.pos()
        self.cfg.window_pos = (pos.x(), pos.y())
        save_config(self.cfg)
        self.hotkeys.stop()
        QtWidgets.QApplication.quit()


def main():
    app = QtWidgets.QApplication(sys.argv)
    # Wispr-ish tooltips for transcript bubble.
    app.setStyleSheet(
        """
        QToolTip {
          background-color: rgba(10, 12, 16, 240);
          color: #F9FAFB;
          border: 1px solid rgba(255,255,255,0.14);
          border-radius: 12px;
          padding: 8px 10px;
          font-size: 12px;
          font-weight: 600;
        }
        """
    )
    _ = App()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
