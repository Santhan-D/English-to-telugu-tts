# English -> Telugu text -> Telugu speech

Models (IndicTrans2, Indic Parler-TTS, Gonthuka ASR) are PyTorch-only, so they run in
`model_server.py`; `pipeline.mjs` (Node 18+, no npm deps) drives them.

## Setup (use a GPU, e.g. Colab T4; CPU works but TTS is slow)
    pip install torch transformers fastapi uvicorn soundfile librosa IndicTransToolkit
    pip install git+https://github.com/huggingface/parler-tts.git
    # IndicTrans2 may be gated: accept its terms on Hugging Face, then `huggingface-cli login`

    SERVICES=translate,tts,asr uvicorn model_server:app --port 8000

If pip versions clash (parler-tts pins transformers), use two virtualenvs:
    SERVICES=translate uvicorn model_server:app --port 8001
    SERVICES=tts,asr   uvicorn model_server:app --port 8002
    TRANSLATE_URL=http://127.0.0.1:8001 TTS_URL=http://127.0.0.1:8002 node pipeline.mjs "Hello"

## Use
    node pipeline.mjs "Please take your medicine now."
    node pipeline.mjs --batch prompts.json --esp     # out/*.te.wav + 16 kHz mono *.16k.wav for ESP32 (needs ffmpeg)
    node pipeline.mjs --asr recording.wav            # Telugu speech -> text

Machine translation can be wrong. Have a Telugu speaker review `out/manifest.json` before flashing medication prompts.
Gonthuka ASR is AGPL-3.0; check that suits how you distribute it.
