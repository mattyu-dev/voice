import argparse
import os
import platform
import sys
import threading
import time
from typing import Optional

import numpy as np
import pyperclip
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard


def _parse_key(s: str):
    s = s.strip().lower()
    if len(s) == 1:
        return keyboard.KeyCode.from_char(s)
    try:
        return getattr(keyboard.Key, s)
    except AttributeError as e:
        raise SystemExit(f"Unsupported key '{s}'. Try e.g. 'f9', 'space', or a single character.") from e


class Recorder:
    def __init__(self, samplerate: int):
        self._samplerate = samplerate
        self._frames = []
        self._lock = threading.Lock()
        self._stream: Optional[sd.InputStream] = None
        self._recording = False

    @property
    def recording(self) -> bool:
        with self._lock:
            return self._recording

    def start(self):
        with self._lock:
            if self._recording:
                return
            self._frames = []
            self._recording = True

        def _callback(indata, frames, time_info, status):
            if status:
                # Avoid noisy logs; keep one-line status in stderr.
                print(f"[audio] {status}", file=sys.stderr)
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


def main():
    # Avoid HF telemetry; nothing here should send user audio anywhere.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    ap = argparse.ArgumentParser(description="Push-to-talk local transcription (copies to clipboard).")
    ap.add_argument("--key", default="f9", help="Push-to-talk key (e.g. f9, space, or a single character).")
    ap.add_argument(
        "--model",
        default="base",
        help="Whisper model name/path (e.g. tiny, base, small; use *.en for English-only).",
    )
    ap.add_argument("--device", default="cpu", help="faster-whisper device: cpu or cuda.")
    ap.add_argument("--compute-type", default="int8", help="e.g. int8 (cpu), float16 (cuda), int8_float16.")
    ap.add_argument("--samplerate", type=int, default=16000, help="Input sample rate (Hz).")
    ap.add_argument("--language", default="auto", help="Language code: auto, en, fr (or any Whisper language).")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--vad", action="store_true", help="Enable VAD filter (can help on long silence).")
    ap.add_argument("--paste", action="store_true", help="After copying, auto-paste into the active app.")
    args = ap.parse_args()

    ptt_key = _parse_key(args.key)
    print(f"Loading model: {args.model} (device={args.device}, compute_type={args.compute_type}) ...")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    rec = Recorder(samplerate=args.samplerate)

    kb_ctl = keyboard.Controller()
    is_macos = platform.system().lower() == "darwin"
    paste_mod = keyboard.Key.cmd if is_macos else keyboard.Key.ctrl

    def do_paste():
        with kb_ctl.pressed(paste_mod):
            kb_ctl.press("v")
            kb_ctl.release("v")

    print("")
    print(f"Hold {args.key.upper()} to record. Release to transcribe. Ctrl+C to exit.")
    print("")

    def on_press(key):
        if key == ptt_key and not rec.recording:
            print("[rec] start")
            try:
                rec.start()
            except Exception as e:
                print(f"[rec] failed to start: {e}", file=sys.stderr)

    def on_release(key):
        if key == ptt_key and rec.recording:
            print("[rec] stop; transcribing ...")
            audio = rec.stop()
            if audio.size == 0:
                print("[asr] no audio captured")
                return

            language = None if args.language.strip().lower() == "auto" else args.language.strip().lower()
            segments, info = model.transcribe(
                audio,
                language=language,
                beam_size=args.beam_size,
                vad_filter=args.vad,
            )
            text = "".join(seg.text for seg in segments).strip()
            if not text:
                print("[asr] empty transcript")
                return

            print("")
            if language is None and getattr(info, "language", None):
                lp = getattr(info, "language_probability", None)
                if lp is None:
                    print(f"[asr] detected language: {info.language}")
                else:
                    print(f"[asr] detected language: {info.language} (p={lp:.2f})")
            print(text)
            print("")

            try:
                pyperclip.copy(text)
                print("[clip] copied")
            except Exception as e:
                print(f"[clip] failed: {e}", file=sys.stderr)

            if args.paste:
                # Give the target app a moment to regain focus after key-up.
                time.sleep(0.05)
                try:
                    do_paste()
                    print("[paste] sent")
                except Exception as e:
                    print(f"[paste] failed: {e}", file=sys.stderr)

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
