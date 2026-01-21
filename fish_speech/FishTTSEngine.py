import os
import re
import sys
import torch
import gc
import shutil
import numpy as np
import soundfile as sf
import platform
from pathlib import Path
from loguru import logger
from datetime import datetime

# --- SYSTEM CONFIGURATION ---
logger.remove()
logger.add(sys.stdout, colorize=True, level="TRACE",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

# Performance optimization for Windows/Linux
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# --- Constants for Directory Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

# --- PRODUCTION PRESETS (YOUR WINNING PARAMETERS) ---
VOICE_PRESETS = {
    "MARLENE": {
        "temp": 0.75,
        "top_p": 0.90,
        "chunk": 900,
        "penalty": 1.10,
        "ref_path": str(PROJECT_ROOT / "voices" / "ElevenLabs_Marlene.mp3"),
        "prompt": """La mente lo es todo. La causa mental. La causa de todo -absolutamente todo- es mental, es decir, 
        la mente es la que produce o causa todo en la vida del individuo.
        Cuando reconozcamos, entendamos y aceptemos esta verdad, habremos dado un paso muy importante en el progreso del desarrollo. 
        Si todo es mental, este es un universo mental, donde todo funciona por medios mentales. Nosotros somos seres 
        mentales, mentalidades buenas, perfectas y eternas.
        La mente sólo tiene una actividad, pensar. El pensamiento es todo lo de la mente lo único que somos y tenemos es 
        pensamiento, por ello, el pensamiento es lo más importante de todo.""",
        "style_tags": "(calm) (deep voice) (slow)"
    },
    "CAMILA": {
        "temp": 0.70,  # FIXED: Lowered from 0.88 to prevent "shrill" voice
        "top_p": 0.70,  # Tighter control to avoid robotic drifting
        "chunk": 900,  # Safe size for Quadro T1000
        "penalty": 1.02,  # Increased to prevent loop/stuttering
        "ref_path": str(PROJECT_ROOT / "voices" / "Camila_Sodi.mp3"),
        "prompt": """Todos venimos de un mismo campo fuente, de una misma gran energía, de un mismo Dios, de un mismo 
        universo, como le quieras llamar. Todos somos parte de eso. Nacemos y nos convertimos en esto por un ratito 
        muy chiquito, muy chiquitito, que creemos que es muy largo y se nos olvida que vamos a regresar a ese lugar 
        de donde venimos, que es lo que tú creas, adonde tú creas, pero inevitablemente vas a regresar.""",
        "style_tags": "(calm) (deep voice) (slow)"
    },
    "ALEJANDRO": {
        "temp": 0.75,
        "top_p": 0.85,
        "chunk": 512,
        "penalty": 1.15,
        "ref_path": str(PROJECT_ROOT / "voices" / "ElevenLabs_Alejandro.mp3"),
        "prompt": """(serious) (calm) La mente lo es todo. La causa mental. La causa de todo -absolutamente todo- es mental, es decir, 
            la mente es la que produce o causa todo en la vida del individuo.
            Cuando reconozcamos, entendamos y aceptemos esta verdad, habremos dado un paso muy importante en el progreso del desarrollo. 
            Si todo es mental, este es un universo mental, donde todo funciona por medios mentales. Nosotros somos seres 
            mentales, mentalidades buenas, perfectas y eternas.
            La mente sólo tiene una actividad, pensar. El pensamiento es todo lo de la mente lo único que somos y tenemos es 
            pensamiento, por ello, el pensamiento es lo más importante de todo.""",
        "style_tags": "(serious) (calm)"
    }
}


class FishTTSEngine:
    def __init__(self, checkpoint_path=None):
        """
        Initializes the S1-Mini engine with memory safeguards.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vocal_dna_cache = {}  # Cache for encoded references

        # Default path to the S1-Mini model (~3.36 GB)
        self.checkpoint_dir = checkpoint_path or (PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini")

        # Use half precision (FP16) for speed and memory efficiency
        self.precision = torch.half

        # Compile only on Linux (Kaggle), disable on Windows to avoid errors
        self.should_compile = False if platform.system() == "Windows" else True

        logger.info(f"🚀 Initializing S1-Mini Engine | Device: {self.device} | Compile: {self.should_compile}")

        try:
            self.engine = self._load_models()
            logger.success("✅ Models loaded successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise e

    def _load_models(self):
        """Loads the Llama queue and VQ-GAN Decoder."""
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=self.checkpoint_dir,
            device=self.device,
            precision=self.precision,
            compile=self.should_compile
        )
        decoder_model = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=self.checkpoint_dir / "codec.pth",
            device=self.device
        )
        return TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=self.precision,
            compile=self.should_compile
        )

    def clean_text(self, text):
        """Sanitizes input text."""
        if not text: return ""
        text = text.replace("\n", " ").replace("\t", " ")
        return re.sub(r'\s+', ' ', text).strip()

    def split_text(self, text, max_chars=600):
        """
        Intelligent splitter using Regex.
        Splits by punctuation (. ! ?) to preserve semantic meaning.
        """
        text = self.clean_text(text)
        # Split by punctuation followed by space
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks, current_chunk = [], ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 < max_chars:
                current_chunk += (sentence + " ")
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def _crossfade_chunks(self, audio_list, crossfade_ms=50, sample_rate=44100):
        """
        Merges audio chunks using a linear crossfade to eliminate robotic clicks.
        """
        if not audio_list: return None
        if len(audio_list) == 1: return audio_list[0]

        fade_samples = int(sample_rate * crossfade_ms / 1000)
        combined = audio_list[0]

        for next_chunk in audio_list[1:]:
            # If chunks are too short, skip fading to avoid index errors
            if len(combined) < fade_samples or len(next_chunk) < fade_samples:
                combined = np.concatenate((combined, next_chunk))
                continue

            # Create fade curves
            fade_out = np.linspace(1, 0, fade_samples)
            fade_in = np.linspace(0, 1, fade_samples)

            # Extract overlapping regions
            tail = combined[-fade_samples:] * fade_out
            head = next_chunk[:fade_samples] * fade_in
            overlap_area = tail + head

            # Stitch: [Body A] + [Overlap] + [Body B]
            combined = np.concatenate((combined[:-fade_samples], overlap_area, next_chunk[fade_samples:]))

        return combined

    def process_narration(self, voice_key, raw_text):
        """
        Main API method. Generates audio, stitches it with crossfade, and normalizes.
        """
        if voice_key not in VOICE_PRESETS:
            logger.error(f"❌ Voice key '{voice_key}' not found.")
            return None, None

        params = VOICE_PRESETS[voice_key]
        clean_input = self.clean_text(raw_text)
        raw_audio_segments = []

        try:
            # --- 1. Vocal DNA Caching ---
            if voice_key in self.vocal_dna_cache:
                audio_bytes, vq_tokens = self.vocal_dna_cache[voice_key]
            else:
                logger.info(f"🧬 Encoding DNA for: {voice_key}")
                with open(params['ref_path'], "rb") as f:
                    audio_bytes = f.read()
                with torch.inference_mode():
                    vq_tokens = self.engine.encode_reference(audio_bytes,
                                                             enable_reference_audio=True
                                                             )
                self.vocal_dna_cache[voice_key] = (audio_bytes,
                                                   vq_tokens
                                                   )

            # --- 2. Chunk Processing ---
            text_chunks = self.split_text(clean_input)
            style_prefix = params.get('style_tags', '')

            for i, chunk_text in enumerate(text_chunks):
                logger.debug(f"⏳ Processing chunk {i + 1}/{len(text_chunks)}")

                # Trick: Add trailing dots to let the model trail off naturally
                processed_text = f"{style_prefix} {chunk_text.strip()} ..."

                req = ServeTTSRequest(
                    text=processed_text,
                    references=[ServeReferenceAudio(
                        audio=audio_bytes,
                        tokens=vq_tokens.tolist(),
                        text=params['prompt']
                    )],
                    max_new_tokens=2048,
                    chunk_length=params['chunk'],
                    top_p=params['top_p'],
                    temperature=params['temp'],
                    repetition_penalty=params['penalty'],
                    format="wav"
                )

                # Inference
                results = self.engine.inference(req)

                # Flatten generator results into a single numpy array for this chunk
                chunk_parts = []
                for res in results:
                    data = res.audio if hasattr(res, 'audio') else res
                    if isinstance(data, np.ndarray):
                        chunk_parts.append(data)
                    elif isinstance(data, tuple):
                        for item in data:
                            if isinstance(item, np.ndarray):
                                chunk_parts.append(item)

                if chunk_parts:
                    # raw_audio_segments.append(np.concatenate(chunk_parts))
                    full_chunk = np.concatenate(chunk_parts)

                    breath_pad = np.zeros(int(44100 * 0.5))
                    full_chunk_with_breath = np.concatenate((full_chunk, breath_pad))

                    raw_audio_segments.append(full_chunk_with_breath)

                # Clean VRAM after each chunk
                torch.cuda.empty_cache()
                gc.collect()

            # --- 3. Post-Processing (Crossfade & Normalize) ---
            if raw_audio_segments:
                logger.info("🔧 Applying Crossfade and Normalization...")

                # Apply Crossfade (50ms overlap)
                final_audio = self._crossfade_chunks(raw_audio_segments, crossfade_ms=50)

                # Soft Limiter / Normalization (Target -1.0 dB)
                max_val = np.abs(final_audio).max()
                if max_val > 0:
                    # Normalize to 0.95 to avoid clipping
                    final_audio = final_audio / max_val * 0.95

                silence_pad = np.zeros(int(44100 * 0.5))
                final_audio = np.concatenate((final_audio, silence_pad))

                return final_audio, 44100

            return None, None

        except Exception as e:
            logger.error(f"🔥 Engine Error: {e}")
            return None, None


# --- TESTING BLOCK ---
if __name__ == "__main__":
    engine = FishTTSEngine()

    LONG_CHAPTER = """
            Todos venimos de un mismo campo fuente, de una misma gran energía, de un mismo Dios, de un mismo 
            universo, como le quieras llamar... Todos somos parte de eso... Nacemos y nos convertimos en esto por un ratito... 
            muy chiquito..., muy chiquitito, que creemos que es muy largo y se nos olvida que vamos a regresar a ese lugar 
            de donde venimos.

            Escucha bien esto. No eres una gota en el océano, eres el océano entero en una gota. Tu imaginación no es un estado 
            de fantasía o ilusión, es la verdadera realidad esperando ser reconocida. Cuando cierras los ojos y asumes el 
            sentimiento de tu deseo cumplido, no estás "fingiendo", estás accediendo a la cuarta dimensión, al mundo de las 
            causas, donde todo ya existe. Lo que ves afuera, en tu mundo físico, es simplemente una pantalla retrasada, un 
            eco de lo que fuiste ayer, de lo que pensaste ayer.

            Si tu realidad actual no te gusta, deja de pelear con la pantalla. No puedes peinar tu reflejo en el espejo, 
            tienes que peinarte tú. Debes cambiar la concepción que tienes de ti mismo. Pregúntate: ¿Quién soy yo ahora? 
            Si la respuesta no es "Soy próspero", "Soy amado", "Soy saludable", entonces estás usando tu poder divino en tu 
            contra. El universo no te juzga, simplemente te dice "SÍ". Si dices "estoy arruinado", el universo dice "SÍ, lo estás". 
            Si dices "Soy abundante", el universo dice "SÍ, lo eres".

            Por lo tanto, el secreto no es el esfuerzo físico ni la lucha externa. El secreto es el cambio interno de estado. 
            Moverte, en tu mente, del estado de carencia al estado de posesión. Sentir la textura de la realidad que deseas 
            hasta que sea tan natural que ya no la busques, porque sabes que ya la tienes. Y cuando esa certeza interna hace 
            clic, el mundo exterior no tiene más remedio que reorganizarse para reflejar tu nueva verdad... Inevitablemente, 
            vas a regresar a tu poder.
        """

    audio_data, sample_rate = engine.process_narration(
        voice_key="CAMILA",
        raw_text=LONG_CHAPTER
    )

    if audio_data is not None:
        output_path = "output_camila_optimized.wav"
        sf.write(output_path, audio_data, sample_rate)
        logger.success(f"🏆 Audio generated successfully: {output_path}")
    else:
        logger.error("❌ Audio generation failed.")