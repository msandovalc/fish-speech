# ==============================================================================
# inference_fish_spanish.py
# ==============================================================================
# Fine-Tuning Lab for Fish Audio S1-Mini — Spanish Voice Cloning.
#
# Purpose:
#   Systematically find the parameter combination that produces the most
#   natural, fluid, human-sounding Spanish TTS from a cloned voice.
#
# Architecture:
#   FishVoiceLab — main class with:
#     - generate()          -> single audio generation (production use)
#     - run_lab()           -> parameter sweep across all axes
#     - run_quick_test()    -> fast 3-combination smoke test
#
# Fixes applied vs original:
#   1. ServeTTSRequest — full parameters re-enabled (temp, top_p, penalty,
#      chunk_length, max_new_tokens, prompt_tokens)
#   2. split_text — paragraph-first strategy (\n\n preserved as pauses)
#   3. Tag injection — every chunk, not just chunk 0
#   4. Silence padding — variable: 80ms phrases, 120ms sentences, 350ms paragraphs
#   5. prompt_tokens — context passed between chunks for rhythmic continuity
#   6. run_lab() — real parameter sweeps across temp, penalty, top_p, tags
# ==============================================================================

import io
import os
import re
import sys
import csv
import torch
import gc
import shutil
import numpy as np
import soundfile as sf
import platform
from pathlib import Path
from loguru import logger
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

# --- SYSTEM CONFIGURATION ---
logger.remove()
logger.add(sys.stdout, colorize=True, level="TRACE",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

PROJECT_ROOT = Path(__file__).resolve().parent

# --- PLATFORM DETECTION ---
# Module-level so it's visible outside the class and applied consistently.
# should_compile=True on Linux (Kaggle GPU) enables torch.compile -> ~10x speedup.
# should_compile=False on Windows — torch.compile requires Triton which is unsupported.
is_windows     = platform.system() == "Windows"
should_compile = False if is_windows else True

# --- IMPORTS WITH FALLBACK ---
try:
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.dac.inference import load_model as load_decoder_model
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
    from fish_speech.utils import set_seed
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT))
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.dac.inference import load_model as load_decoder_model
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
    from fish_speech.utils import set_seed

# ==============================================================================
# VOICE PRESETS
# Each preset defines the reference audio + transcription for voice cloning.
# Style tags are the DEFAULT for production — lab sweeps override them.
# ==============================================================================
VOICE_PRESETS = {
    "CRISTY": {
        "temp":    0.70,
        "top_p":   0.70,
        "chunk":   300,
        "penalty": 1.035,
        "ref_path": str(PROJECT_ROOT / "voices" / "cristy_de_la_torre.wav"),
        "prompt": (
            "Todo es mente, el universo es mental y Jesus lo dijo sin rodeos, Jesus decia si tuvieras fe como "
            "un grano de mostaza, le dirias a la montana, muevete y la montana se moveria. Esto no es metafora poetica, "
            "esto es ingenieria de conciencia, la fe no es creer algo bonito, la fe es alinear tu mente con la mente "
            "divina, con la mente universal, por eso Jesus buscaba la soledad, porque los milagros no se dan en el ruido, "
            "se dan en el silencio, la preocupacion es caos mental, la fe es coherencia. Preguntate esto sin huir, "
            "que pienso todos los dias, que mundo esta creando mi mente. Jesus y Hermes lo sabian, el mundo responde a "
            "la calidad de tu atencion, por eso donde pones tu atencion pones tu energia."
        ),
        "style_tags": "(empathetic)(soft tone)",
    },
}


# ==============================================================================
# LAB CONFIG — Parameter search space
# Edit these lists to define what the lab sweeps over.
# ==============================================================================
@dataclass
class LabConfig:
    # Temperature — controls creativity / naturalness
    # 0.60-0.65: very stable, consistent, slightly robotic
    # 0.70-0.75: balanced — good for voice cloning
    # 0.80-0.85: more expressive, risk of artifacts above 0.82
    temperatures: List[float] = field(default_factory=lambda: [0.65, 0.70, 0.75, 0.80])

    # Top-p — nucleus sampling cutoff
    # 0.70: conservative, stable
    # 0.80: more expressive prosody
    # 0.90: highest variation, slightly less predictable
    top_p_values: List[float] = field(default_factory=lambda: [0.70, 0.80, 0.90])

    # Repetition penalty — prevents loops and stuttering
    # 1.00: no penalty (risky for long text)
    # 1.035: safe default
    # 1.05: moderate
    # 1.10: aggressive — good for penalty-prone voices
    penalties: List[float] = field(default_factory=lambda: [1.02, 1.035, 1.05, 1.10])

    # Style tags — emotional tone injection per chunk
    tag_variants: List[str] = field(default_factory=lambda: [
        "(empathetic)(soft tone)",     # warm, grounded — dark stoic baseline
        "(empathetic)(narrator)",      # narrative distance with warmth
        "(sincere)(soft tone)",        # direct, intimate
        "(concerned)(soft tone)",      # careful, weighted
        "(comforting)",                # soothing, resolution energy
        "",                            # no tags — pure reference cloning
    ])

    # Chunk size — affects prosody grouping per inference call
    # 200: more frequent style refresh, slightly more joins
    # 300: more natural flow per chunk, more VRAM
    chunk_sizes: List[int] = field(default_factory=lambda: [200, 300])

    # Fixed seed for reproducibility across all combos
    seed: int = 1234


# ==============================================================================
# MAIN CLASS
# ==============================================================================
class FishVoiceLab:
    """
    Fine-Tuning Lab for Fish Audio S1-Mini voice cloning.

    Three entry points:
      lab.generate(voice_key, text)         -> production single generation
      lab.run_lab(text, voice_key)          -> full parameter sweep
      lab.run_quick_test(text, voice_key)   -> fast 3-combo smoke test
    """

    def __init__(self, checkpoint_path: Optional[Path] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vocal_dna_cache = {}

        self.checkpoint_dir = checkpoint_path or (
            PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini"
        )
        self.precision = torch.half
        self.should_compile = should_compile  # module-level: True on Kaggle/Linux, False on Windows

        logger.info(
            f"Initializing S1-Mini | Device: {self.device} | "
            f"Compile: {self.should_compile}"
        )
        self.engine = self._load_models()
        torch.cuda.empty_cache()
        gc.collect()
        logger.success("Engine ready.")

    # =========================================================================
    # ENGINE SETUP
    # =========================================================================
    def _load_models(self) -> TTSInferenceEngine:
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=self.checkpoint_dir,
            device=self.device,
            precision=self.precision,
            compile=self.should_compile,
        )
        decoder_model = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=self.checkpoint_dir / "codec.pth",
            device=self.device,
        )
        return TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=self.precision,
            compile=self.should_compile,
        )

    def _load_reference(self, voice_key: str, max_duration: int = 60) -> bytes:
        """Loads and caches reference audio bytes for voice cloning."""
        params = VOICE_PRESETS[voice_key]
        cache_key = (voice_key, params["ref_path"])

        if cache_key in self.vocal_dna_cache:
            return self.vocal_dna_cache[cache_key]

        try:
            data, sr = sf.read(params["ref_path"])
            if len(data) > sr * max_duration:
                logger.warning(f"Reference too long — trimming to {max_duration}s.")
                data = data[: int(sr * max_duration)]
            buf = io.BytesIO()
            sf.write(buf, data, sr, format="WAV")
            audio_bytes = buf.getvalue()
        except Exception as e:
            logger.error(f"Error loading reference: {e}")
            with open(params["ref_path"], "rb") as f:
                audio_bytes = f.read()

        self.vocal_dna_cache[cache_key] = audio_bytes
        return audio_bytes

    # =========================================================================
    # TEXT PROCESSING
    # =========================================================================
    def _clean_text(self, text: str) -> str:
        """Sanitizes a single paragraph. Does NOT strip newlines."""
        if not text:
            return ""
        text = re.sub(r"([.!?])(?=\S)", r"\1 ", text)
        text = text.replace("\t", " ")
        return re.sub(r"\s+", " ", text).strip()

    def _split_text(self, text: str, max_chars: int = 200) -> List[str]:
        """
        PARAGRAPH-FIRST split strategy.

        Step 1: Split by double newlines (\\n\\n) FIRST — before any cleaning
                that strips newlines. This preserves paragraph boundaries as
                natural pause points.
        Step 2: If a paragraph exceeds max_chars, split by sentence internally.

        Why this matters:
        The 4-paragraph dark stoic script (Hook / Truth / Resolution / CTA)
        has distinct emotional weight per paragraph. Flattening them loses
        all natural pause and rhythm structure.
        """
        if not text:
            return []

        chunks = []
        paragraphs = re.split(r"\n\n+", text.strip())

        for paragraph in paragraphs:
            clean = self._clean_text(paragraph)
            if not clean:
                continue

            if len(clean) <= max_chars:
                chunks.append(clean)
            else:
                # Paragraph too long — split by sentence
                sentences = re.split(r"(?<=[.!?])\s+", clean)
                current = ""
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    if len(current) + len(sentence) < max_chars:
                        current += sentence + " "
                    else:
                        if current:
                            chunks.append(current.strip())
                        current = sentence + " "
                if current:
                    chunks.append(current.strip())

        return chunks

    def _inject_tag(
        self,
        chunk_text: str,
        chunk_idx: int,
        style_tags: str,
        paragraph_tags: Optional[List[str]] = None,
    ) -> str:
        """
        Injects emotion markers at the beginning of each chunk.

        Rules (per Fish Audio S1 official docs):
        - Tags MUST go at the beginning of sentences.
        - If chunk already starts with a marker like "(concerned)", pass through.
          Do NOT double-inject — model reads duplicates as literal text.
        - If paragraph_tags provided, rotate cyclically P1 P2 P3 P4 P1...
        - If only style_tags, inject on EVERY chunk to prevent style drift.
        """
        already_tagged = bool(re.match(r"^\([a-z\s\-]+\)", chunk_text.strip()))
        if already_tagged:
            return chunk_text

        if paragraph_tags:
            tag = paragraph_tags[chunk_idx % len(paragraph_tags)]
            return f"{tag} {chunk_text}".strip()

        if style_tags:
            return f"{style_tags} {chunk_text}".strip()

        return chunk_text

    def _variable_silence(
        self, chunk_text: str, sample_rate: int, dtype
    ) -> np.ndarray:
        """
        Returns silence padding that mimics natural human speech rhythm.

        Human speech pauses by context:
        - Short phrase (< 80 chars):    ~80ms
        - Sentence ending in '.':       ~120ms
        - Sentence ending in '?' / '!': ~150ms
        - Paragraph-level chunk:        ~350ms

        Fixed 250ms after every chunk (the original approach) creates a
        mechanical, metronomic rhythm that sounds robotic.
        """
        text = chunk_text.strip()
        char_count = len(text)

        if char_count > 150:
            ms = 350
        elif text.endswith("."):
            ms = 120
        elif text.endswith(("?", "!")):
            ms = 150
        else:
            ms = 80

        samples = int(sample_rate * ms / 1000)
        return np.zeros(samples, dtype=dtype)

    def _crossfade(
        self,
        audio_list: List[np.ndarray],
        crossfade_ms: int = 30,
        sample_rate: int = 44100,
    ) -> Optional[np.ndarray]:
        """Linear crossfade between chunks to eliminate robotic clicks at joins."""
        if not audio_list:
            return None
        if len(audio_list) == 1:
            return audio_list[0]

        fade_samples = int(sample_rate * crossfade_ms / 1000)
        combined = audio_list[0]

        for next_chunk in audio_list[1:]:
            if len(combined) < fade_samples or len(next_chunk) < fade_samples:
                combined = np.concatenate((combined, next_chunk))
                continue
            fade_out = np.linspace(1, 0, fade_samples)
            fade_in = np.linspace(0, 1, fade_samples)
            tail = combined[-fade_samples:] * fade_out
            head = next_chunk[:fade_samples] * fade_in
            overlap = tail + head
            combined = np.concatenate(
                (combined[:-fade_samples], overlap, next_chunk[fade_samples:])
            )

        return combined

    def _normalize(self, audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
        max_val = np.abs(audio).max()
        if max_val == 0:
            return audio
        return audio * (10 ** (target_db / 20) / max_val)

    # =========================================================================
    # CORE GENERATION
    # =========================================================================
    def generate(
        self,
        voice_key: str,
        raw_text: str,
        temp: Optional[float] = None,
        top_p: Optional[float] = None,
        penalty: Optional[float] = None,
        chunk_size: Optional[int] = None,
        style_tags: Optional[str] = None,
        paragraph_tags: Optional[List[str]] = None,
        seed: int = 1234,
        max_chars: int = 200,
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Core generation. All parameters optional — falls back to preset values.

        Args:
            voice_key:       Key from VOICE_PRESETS.
            raw_text:        Input text. May contain \\n\\n paragraph breaks.
            temp:            Temperature (0.60-0.85). Lower=stable, higher=natural.
            top_p:           Nucleus sampling (0.70-0.95).
            penalty:         Repetition penalty (1.0-1.15).
            chunk_size:      Max tokens per inference call (200-400).
            style_tags:      Emotion tags injected on every chunk.
            paragraph_tags:  Per-paragraph tags list — overrides style_tags.
            seed:            Reproducibility seed.
            max_chars:       Max chars per text chunk before splitting.

        Returns:
            (audio_array, sample_rate) or (None, 0) on failure.
        """
        if voice_key not in VOICE_PRESETS:
            logger.error(f"Voice '{voice_key}' not found in VOICE_PRESETS.")
            return None, 0

        params = VOICE_PRESETS[voice_key]

        _temp    = temp       if temp       is not None else params["temp"]
        _top_p   = top_p      if top_p      is not None else params["top_p"]
        _penalty = penalty    if penalty    is not None else params["penalty"]
        _chunk   = chunk_size if chunk_size is not None else params["chunk"]
        _tags    = style_tags if style_tags is not None else params.get("style_tags", "")
        _ptags   = paragraph_tags

        set_seed(seed)
        audio_bytes = self._load_reference(voice_key)
        text_chunks = self._split_text(raw_text, max_chars=max_chars)

        logger.info(
            f"Generating | {voice_key} | {len(text_chunks)} chunks | "
            f"T={_temp} top_p={_top_p} pen={_penalty} chunk={_chunk} | tags='{_tags}'"
        )

        raw_segments = []
        hist_tokens  = None
        hist_text    = None

        try:
            for i, chunk_text in enumerate(text_chunks):
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue

                processed_text = self._inject_tag(chunk_text, i, _tags, _ptags)
                logger.debug(f"  Chunk {i + 1}/{len(text_chunks)}: {processed_text[:70]}...")

                best_attempt = None
                final_res    = None

                for attempt in range(3):
                    if attempt > 0:
                        set_seed(seed + i + attempt * 100)

                    req = ServeTTSRequest(
                        text=processed_text,
                        references=[
                            ServeReferenceAudio(
                                audio=audio_bytes,
                                text=params["prompt"],
                            )
                        ],
                        use_memory_cache="on",
                        chunk_length=_chunk,
                        max_new_tokens=1024,
                        top_p=_top_p,
                        temperature=_temp,
                        repetition_penalty=_penalty,
                        format="wav",
                        # Passing previous chunk context improves rhythmic continuity
                        # between chunks — the model "remembers" the pace and tone.
                        prompt_text=[hist_text]     if hist_text   is not None else None,
                        prompt_tokens=[hist_tokens] if hist_tokens is not None else None,
                    )

                    final_res = None
                    for res in self.engine.inference(req):
                        if res.code == "final":
                            final_res = res
                            break

                    if final_res and final_res.codes is not None:
                        num_tokens = final_res.codes.shape[1]
                        if num_tokens < len(chunk_text):
                            logger.warning(
                                f"Chunk {i + 1} too short "
                                f"({num_tokens} tokens vs {len(chunk_text)} chars). Retrying..."
                            )
                            continue
                        best_attempt = final_res
                        break

                if best_attempt is None and final_res is not None:
                    logger.warning(f"Chunk {i + 1}: all retries failed, using last result.")
                    best_attempt = final_res

                if best_attempt is None or best_attempt.audio is None:
                    logger.warning(f"Chunk {i + 1}: no audio generated, skipping.")
                    continue

                sr, audio_np = best_attempt.audio

                # Variable silence — natural rhythm (not fixed 250ms)
                silence = self._variable_silence(chunk_text, sr, audio_np.dtype)
                raw_segments.append(np.concatenate((audio_np, silence)))

                # Update context WITH tags — preserves emotional continuity
                if best_attempt.codes is not None:
                    codes = torch.from_numpy(best_attempt.codes).to(torch.int)
                    keep = 50  # ~1s of context
                    if codes.shape[1] > keep:
                        codes = codes[:, -keep:]
                    hist_tokens = codes
                    hist_text   = processed_text  # WITH tags for context

                torch.cuda.empty_cache()
                gc.collect()

            if not raw_segments:
                logger.error("No audio segments generated.")
                return None, 0

            merged      = self._crossfade(raw_segments, crossfade_ms=30)
            final_audio = self._normalize(merged, target_db=-1.0)
            final_audio = np.concatenate((final_audio, np.zeros(int(44100 * 0.5))))

            return final_audio, 44100

        except Exception as e:
            logger.error(f"Engine error: {e}")
            import traceback
            traceback.print_exc()
            return None, 0

    # =========================================================================
    # FINE-TUNING LAB
    # =========================================================================
    def run_lab(
        self,
        text: str,
        voice_key: str = "CRISTY",
        config: Optional[LabConfig] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        """
        Full parameter sweep lab.

        Generates one audio file per parameter combination and writes a
        CSV report. Open results.csv, listen to each file, fill in the
        'rating' column (1-5), then copy the best params to VOICE_PRESETS.

        Output structure:
            LAB_CRISTY_143022/
              CRISTY_T0.65_P0.70_PEN1.035_CH200_empathetic_soft_tone.wav
              CRISTY_T0.70_P0.80_PEN1.050_CH300_sincere.wav
              ...
              results.csv
        """
        cfg = config or LabConfig()
        timestamp = datetime.now().strftime("%H%M%S")
        folder = output_dir or (PROJECT_ROOT / f"LAB_{voice_key}_{timestamp}")
        folder.mkdir(parents=True, exist_ok=True)

        combos = [
            (temp, top_p, penalty, chunk_size, tags)
            for temp       in cfg.temperatures
            for top_p      in cfg.top_p_values
            for penalty    in cfg.penalties
            for chunk_size in cfg.chunk_sizes
            for tags       in cfg.tag_variants
        ]

        logger.info(
            f"Lab: {voice_key} | {len(combos)} combinations | Output: {folder}"
        )

        csv_rows = []

        for idx, (temp, top_p, penalty, chunk_size, tags) in enumerate(combos):
            tag_slug = (
                tags.replace("(", "").replace(")", "")
                    .replace(" ", "_").strip("_")
                or "no_tags"
            )
            filename = (
                f"{voice_key}_T{temp}_P{top_p}_PEN{penalty}_"
                f"CH{chunk_size}_{tag_slug}.wav"
            )

            logger.info(
                f"[{idx + 1}/{len(combos)}] "
                f"T={temp} top_p={top_p} pen={penalty} "
                f"chunk={chunk_size} tags='{tags}'"
            )

            audio, sr = self.generate(
                voice_key=voice_key,
                raw_text=text,
                temp=temp,
                top_p=top_p,
                penalty=penalty,
                chunk_size=chunk_size,
                style_tags=tags,
                seed=cfg.seed,
            )

            if audio is not None:
                sf.write(str(folder / filename), audio, sr, subtype="PCM_16")
                duration = round(len(audio) / sr, 2)
                logger.success(f"{filename} ({duration}s)")
                csv_rows.append({
                    "file": filename, "temp": temp, "top_p": top_p,
                    "penalty": penalty, "chunk_size": chunk_size, "tags": tags,
                    "duration_s": duration, "rating": "", "notes": "",
                })
            else:
                logger.error(f"Failed: {filename}")
                csv_rows.append({
                    "file": filename, "temp": temp, "top_p": top_p,
                    "penalty": penalty, "chunk_size": chunk_size, "tags": tags,
                    "duration_s": 0, "rating": "FAILED", "notes": "",
                })

        # Write CSV report for rating
        csv_path = folder / "results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "file", "temp", "top_p", "penalty",
                "chunk_size", "tags", "duration_s", "rating", "notes"
            ])
            writer.writeheader()
            writer.writerows(csv_rows)

        logger.success(
            f"Lab complete. {len(csv_rows)} files in {folder}\n"
            f"   Open results.csv — listen to each file, fill in 'rating' (1-5).\n"
            f"   Best config -> copy those params to VOICE_PRESETS."
        )

        zip_path = PROJECT_ROOT / f"LAB_{voice_key}_{timestamp}"
        shutil.make_archive(str(zip_path), "zip", folder)
        logger.success(f"ZIP ready: {zip_path}.zip")

    def run_quick_test(
        self,
        text: str,
        voice_key: str = "CRISTY",
        output_dir: Optional[Path] = None,
    ) -> None:
        """
        Fast smoke test — 3 key combinations covering the main quality axes.

        Use this first to verify the engine and get a quick baseline
        before committing to the full lab sweep.

        Combo 1 — Conservative: low temp, no tags (pure reference cloning)
        Combo 2 — Balanced:     mid temp, empathetic+soft tone (dark stoic default)
        Combo 3 — Expressive:   higher temp + top_p (more natural variation)
        """
        quick_combos = [
            (0.65, 0.70, 1.035, 300, "",                         "conservative_no_tags"),
            (0.70, 0.70, 1.035, 300, "(empathetic)(soft tone)",  "balanced_dark_stoic"),
            (0.75, 0.85, 1.05,  300, "(sincere)(soft tone)",     "expressive_sincere"),
        ]

        timestamp = datetime.now().strftime("%H%M%S")
        folder = output_dir or (PROJECT_ROOT / f"QUICKTEST_{voice_key}_{timestamp}")
        folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Quick test: {voice_key} | 3 combinations | {folder}")

        for temp, top_p, penalty, chunk, tags, label in quick_combos:
            filename = f"{voice_key}_{label}.wav"
            logger.info(f"  -> {label}: T={temp} top_p={top_p} pen={penalty} tags='{tags}'")

            audio, sr = self.generate(
                voice_key=voice_key,
                raw_text=text,
                temp=temp,
                top_p=top_p,
                penalty=penalty,
                chunk_size=chunk,
                style_tags=tags,
            )

            if audio is not None:
                out_path = folder / filename
                sf.write(str(out_path), audio, sr, subtype="PCM_16")
                logger.success(f"  {filename} ({len(audio) / sr:.1f}s)")
            else:
                logger.error(f"  Failed: {label}")

        logger.success(f"Quick test done. Listen to 3 files in: {folder}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":

    lab = FishVoiceLab()

    # Plain text without pre-embedded tags (for lab tag sweep)
    PLAIN_TEST = """Llevas anos sintiendote exhausta. Y ya ni recuerdas cuando no fue asi. Eso que sientes al despertar no es tu edad. Es tu cuerpo diciendote que algo no esta bien.

No es el trabajo. Es que llevas anos cargando lo que no te corresponde. Tu sistema nervioso aprendio que descansar era peligroso, que si parabas, algo se rompia. Y asi llevas, sosteniendolo todo, para todos, menos para ti.

Eso puede soltarse. No de golpe, pero puede. No tienes que seguir cargando lo que no te pidieron que cargaras. Hay una salida que no implica seguir aguantando.

Comenta la palabra CALMA y te mando el protocolo."""

    # Dark stoic text WITH pre-embedded markers (for testing marker pass-through)
    TAGGED_TEST = """(concerned) Llevas anos sintiendote exhausta. (break) Y ya ni recuerdas cuando no fue asi. Eso que sientes al despertar no es tu edad. Es tu cuerpo diciendote que algo no esta bien.

(empathetic) No es el trabajo. Es que llevas anos cargando lo que no te corresponde. Tu sistema nervioso aprendio que descansar era peligroso. Y asi llevas, sosteniendolo todo, para todos, menos para ti.

(comforting) Eso puede soltarse. No de golpe, pero puede. No tienes que seguir cargando lo que no te pidieron que cargaras. Hay una salida que no implica seguir aguantando.

(sincere) Comenta la palabra CALMA y te mando el protocolo."""

    # ==========================================================================
    # CHOOSE YOUR MODE — uncomment the one you want to run
    # ==========================================================================

    # MODE 1 — Quick test (3 files, ~5 min) — START HERE
    lab.run_quick_test(PLAIN_TEST, voice_key="CRISTY")

    # MODE 2 — Full lab sweep
    # Customize config to focus on specific axes
    # lab.run_lab(
    #     PLAIN_TEST,
    #     voice_key="CRISTY",
    #     config=LabConfig(
    #         temperatures=[0.65, 0.70, 0.75],
    #         top_p_values=[0.70, 0.80],
    #         penalties=[1.035, 1.05],
    #         chunk_sizes=[300],
    #         tag_variants=[
    #             "(empathetic)(soft tone)",
    #             "(sincere)(soft tone)",
    #             "(concerned)(soft tone)",
    #             "",
    #         ],
    #     ),
    # )

    # MODE 3 — Single production generation with best-known params
    # audio, sr = lab.generate(
    #     voice_key="CRISTY",
    #     raw_text=TAGGED_TEST,
    #     temp=0.70,
    #     top_p=0.70,
    #     penalty=1.035,
    #     chunk_size=300,
    #     style_tags="(empathetic)(soft tone)",
    # )
    # if audio is not None:
    #     sf.write("output_production.wav", audio, sr, subtype="PCM_16")
    #     logger.success(f"Production audio: {len(audio) / sr:.1f}s")