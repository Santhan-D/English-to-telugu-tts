#!/usr/bin/env node
// English -> Telugu text -> Telugu speech, plus Telugu speech -> text.
// Node 18+, no npm packages. Talks to model_server.py.
//
//   node pipeline.mjs "Please take your medicine now."
//   node pipeline.mjs --batch prompts.json [--esp] [--force]
//   node pipeline.mjs --asr recording.wav
import { readFile, writeFile, mkdir, access } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { pathToFileURL } from "node:url";
import path from "node:path";

const BASE = process.env.MODEL_SERVER ?? "http://127.0.0.1:8000";
const URLS = {
  translate: process.env.TRANSLATE_URL ?? BASE,
  tts: process.env.TTS_URL ?? BASE,
  asr: process.env.ASR_URL ?? BASE,
};

async function call(service, route, init) {
  const res = await fetch(`${URLS[service]}${route}`, init);
  if (!res.ok) throw new Error(`${service} ${route} failed: ${res.status} ${await res.text()}`);
  return res;
}
const postJson = (body) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export async function translate(text) {
  const res = await call("translate", "/translate", postJson({ text }));
  return (await res.json()).telugu;
}

export async function speak(teluguText, { speaker = "Prakash", pace = "slightly slow" } = {}) {
  const res = await call("tts", "/tts", postJson({ text: teluguText, speaker, pace }));
  return Buffer.from(await res.arrayBuffer());
}

export async function transcribe(wavBuffer) {
  const res = await call("asr", "/asr", {
    method: "POST",
    headers: { "Content-Type": "audio/wav" },
    body: wavBuffer,
  });
  return (await res.json()).text;
}

export async function englishToTeluguAudio(text, opts) {
  const telugu = await translate(text);
  const wav = await speak(telugu, opts);
  return { telugu, wav };
}

// 16 kHz mono 16-bit WAV: small enough for ESP32 LittleFS and easy to play via I2S.
function toEsp(wavPath) {
  const out = wavPath.replace(/\.wav$/, ".16k.wav");
  const r = spawnSync("ffmpeg", ["-y", "-loglevel", "error", "-i", wavPath, "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", out]);
  if (r.error || r.status !== 0) {
    console.warn("ffmpeg missing or failed; skipping ESP32 conversion");
    return null;
  }
  return out;
}

const exists = (p) => access(p).then(() => true, () => false);

async function runBatch(file, { force, esp }) {
  const items = JSON.parse(await readFile(file, "utf8"));
  await mkdir("out", { recursive: true });
  const manifest = [];
  for (const { id, text } of items) {
    const wavPath = path.join("out", `${id}.te.wav`);
    const txtPath = path.join("out", `${id}.te.txt`);
    let telugu;
    if (!force && (await exists(wavPath)) && (await exists(txtPath))) {
      telugu = await readFile(txtPath, "utf8");
      console.log(`skip ${id} (cached)`);
    } else {
      const r = await englishToTeluguAudio(text);
      telugu = r.telugu;
      await writeFile(txtPath, telugu);
      await writeFile(wavPath, r.wav);
      console.log(`${id}: ${text} -> ${telugu}`);
    }
    manifest.push({ id, english: text, telugu, wav: wavPath, esp: esp ? toEsp(wavPath) : null, reviewed: false });
  }
  await writeFile("out/manifest.json", JSON.stringify(manifest, null, 2));
  console.log("Wrote out/manifest.json. Have a Telugu speaker review every line before flashing.");
}

async function main() {
  const args = process.argv.slice(2);
  const flag = (n) => args.includes(n);
  const val = (n) => (args.includes(n) ? args[args.indexOf(n) + 1] : undefined);

  if (val("--asr")) {
    console.log(await transcribe(await readFile(val("--asr"))));
  } else if (val("--batch")) {
    await runBatch(val("--batch"), { force: flag("--force"), esp: flag("--esp") });
  } else {
    const text = args.filter((a) => !a.startsWith("--")).join(" ");
    if (!text) {
      console.error('Usage: node pipeline.mjs "English text" | --batch prompts.json [--esp] [--force] | --asr file.wav');
      process.exit(1);
    }
    const { telugu, wav } = await englishToTeluguAudio(text);
    await mkdir("out", { recursive: true });
    const slug = text.toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 40);
    const file = path.join("out", `${slug}.te.wav`);
    await writeFile(file, wav);
    console.log(`${text}\n-> ${telugu}\n-> ${file}`);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((e) => {
    console.error(e.message);
    process.exit(1);
  });
}
