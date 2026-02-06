# voice

Minimal push-to-talk local transcription (Windows + macOS).

Goal: hold a hotkey, speak, release to transcribe. Output can either paste into the active app (Wispr-like) or copy to clipboard. Audio stays local.

## Prereqs

1. Windows 11 or macOS 13+
2. Python 3.10+

Notes:
- On macOS you may need to grant Microphone permission, and Input Monitoring permission for global hotkeys.
- If `sounddevice` install fails on macOS, try `brew install portaudio` and reinstall.

## 1) Install deps

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## 2) Run push-to-talk

```powershell
.\.venv\Scripts\Activate.ps1
python .\voice_gui.py
```

Hold your hotkey (default `F9`) to record, release to transcribe.

For best results:
- English-only: use `--model base.en` (smaller/faster)
- English + French: use `--model base` (multilingual)

## Settings

Right-click the Flow bar or use the tray icon to open Settings:
- Hotkey (push-to-talk)
- Model (tiny/base/small, multilingual or `.en`)
- Language (auto/en/fr)
- Output (paste into active app or copy to clipboard)
- Preserve clipboard when pasting
- Remember last dragged position (otherwise bottom-center)

## Privacy

- This repo runs transcription locally.
- Only model downloads require network access. After the model is downloaded, you can run offline.

## Build a Windows .exe

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
pyinstaller --noconfirm --onefile --windowed --name Voice .\voice_gui.py
```

The built executable will be in `.\dist\Voice.exe`.
