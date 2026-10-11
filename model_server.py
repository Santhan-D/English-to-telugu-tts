"""Model server for the Telugu voice pipeline.

  POST /translate  {"text": "..."}                    -> {"telugu": "..."}      (IndicTrans2, en -> te)
  POST /tts        {"text": "<telugu>", "speaker"...} -> audio/wav              (Indic Parler-TTS)
  POST /asr        <raw wav bytes>                    -> {"text": "<telugu>"}   (Swecha Gonthuka ASR)

Run:  SERVICES=translate,tts,asr uvicorn model_server:app --port 8000
Use SERVICES to run only some models per process (handy if pip versions conflict).
"""
import io
import os
import threading

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

SERVICES = set(os.getenv("SERVICES", "translate,tts,asr").split(","))
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MT_ID = os.getenv("MT_MODEL", "ai4bharat/indictrans2-en-indic-dist-200M")
TTS_ID = os.getenv("TTS_MODEL", "naklitechie/indic-parler-tts")  # ungated mirror of ai4bharat/indic-parler-tts
ASR_ID = os.getenv("ASR_MODEL", "swechatelangana/swecha-gonthuka-asr")

app = FastAPI(title="Telugu voice pipeline")
lock = threading.Lock()  # one generation at a time


class Req(BaseModel):
    text: str
    speaker: str = "Prakash"  # recommended Telugu voices: Prakash, Lalitha
    pace: str = "slightly slow"


if "translate" in SERVICES:
    try:
        from IndicTransToolkit.processor import IndicProcessor
    except ImportError:
        from IndicTransToolkit import IndicProcessor
    mt_tok = AutoTokenizer.from_pretrained(MT_ID, trust_remote_code=True)
    mt_model = AutoModelForSeq2SeqLM.from_pretrained(MT_ID, trust_remote_code=True).to(DEVICE).eval()
    ip = IndicProcessor(inference=True)

if "tts" in SERVICES:
    from parler_tts import ParlerTTSForConditionalGeneration

    tts_model = ParlerTTSForConditionalGeneration.from_pretrained(TTS_ID).to(DEVICE)
    tts_tok = AutoTokenizer.from_pretrained(TTS_ID)
    desc_tok = AutoTokenizer.from_pretrained(tts_model.config.text_encoder._name_or_path)

if "asr" in SERVICES:
    import librosa

    asr = pipeline("automatic-speech-recognition", model=ASR_ID, device=DEVICE)


def need(name: str):
    if name not in SERVICES:
        raise HTTPException(503, f"'{name}' is not enabled on this server (SERVICES={sorted(SERVICES)})")


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "services": sorted(SERVICES)}


@app.post("/translate")
def translate(body: Req):
    need("translate")
    batch = ip.preprocess_batch([body.text], src_lang="eng_Latn", tgt_lang="tel_Telu")
    with lock, torch.no_grad():
        inputs = mt_tok(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
        out = mt_model.generate(**inputs, use_cache=True, max_length=256, num_beams=5, num_return_sequences=1)
    decoded = mt_tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return {"telugu": ip.postprocess_batch(decoded, lang="tel_Telu")[0]}


@app.post("/tts")
def tts(body: Req):
    need("tts")
    description = (
        f"{body.speaker}'s voice is clear and {body.pace} in delivery, with a very close "
        "recording that has no background noise. Very clear audio."
    )
    d = desc_tok(description, return_tensors="pt").to(DEVICE)
    p = tts_tok(body.text, return_tensors="pt").to(DEVICE)
    with lock, torch.no_grad():
        gen = tts_model.generate(
            input_ids=d.input_ids,
            attention_mask=d.attention_mask,
            prompt_input_ids=p.input_ids,
            prompt_attention_mask=p.attention_mask,
        )
    buf = io.BytesIO()
    sf.write(buf, gen.cpu().numpy().squeeze(), tts_model.config.sampling_rate, format="WAV")
    return Response(buf.getvalue(), media_type="audio/wav")


@app.post("/asr")
async def transcribe(request: Request):
    need("asr")
    audio, _ = librosa.load(io.BytesIO(await request.body()), sr=16000, mono=True)
    with lock:
        text = asr({"raw": audio, "sampling_rate": 16000})["text"]
    return {"text": text}
