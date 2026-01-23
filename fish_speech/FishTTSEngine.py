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
        "temp": 0.65,
        "top_p": 0.90,
        "chunk": 300,
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
        "temp": 0.65,  # FIXED: Lowered from 0.88 to prevent "shrill" voice
        "top_p": 0.70,  # Tighter control to avoid robotic drifting
        "chunk": 300,  # Safe size for Quadro T1000
        "penalty": 1.035,  # Increased to prevent loop/stuttering
        "ref_path": str(PROJECT_ROOT / "voices" / "Camila_Sodi.mp3"),
        "prompt": """Todos venimos de un mismo campo fuente, de una misma gran energía, de un mismo Dios, de un mismo 
        universo, como le quieras llamar. Todos somos parte de eso. Nacemos y nos convertimos en esto por un ratito 
        muy chiquito, muy chiquitito, que creemos que es muy largo y se nos olvida que vamos a regresar a ese lugar 
        de donde venimos, que es lo que tú creas, adonde tú creas, pero inevitablemente vas a regresar.""",
        "style_tags": "(calm) (narrator)"
    },
    "ALEJANDRO": {
        "temp": 0.65,
        "top_p": 0.85,
        "chunk": 300,
        "penalty": 1.15,
        "ref_path": str(PROJECT_ROOT / "voices" / "ElevenLabs_Alejandro.mp3"),
        "prompt": """(serious) (calm) La mente lo es todo. La causa mental. La causa de todo -absolutamente todo- es mental, es decir, 
            la mente es la que produce o causa todo en la vida del individuo.
            Cuando reconozcamos, entendamos y aceptemos esta verdad, habremos dado un paso muy importante en el progreso del desarrollo. 
            Si todo es mental, este es un universo mental, donde todo funciona por medios mentales. Nosotros somos seres 
            mentales, mentalidades buenas, perfectas y eternas.
            La mente sólo tiene una actividad, pensar. El pensamiento es todo lo de la mente lo único que somos y tenemos es 
            pensamiento, por ello, el pensamiento es lo más importante de todo.""",
        "style_tags": "(calm) (narrator)"
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

        torch.cuda.empty_cache()
        gc.collect()

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

    def split_text(self, text, max_chars=400):
        """
        HYBRID SPLIT STRATEGY (PARAGRAPHS + FATIGUE CONTROL):

        1. Primary Logic: Split by visual paragraphs (double newlines).
        2. Secondary Logic (Fatigue Check): If a paragraph is longer than 'max_chars'
           (e.g., 400), it forces an internal split by sentences.

        Why?
        Long text blocks cause 'Style Drift' (loss of tone) and hallucinations
        at the end. By forcing a split on long paragraphs, we refresh the
        style tags "(calm) (deep voice)" more frequently, keeping the voice stable.

        Args:
            text (str): Input text.
            max_chars (int): The safety limit. If a paragraph exceeds this,
                             it gets chopped. Recommended: 400-450.
        """
        # Clean up input text
        text = self.clean_text(text)
        sentences = re.split(r'(?<=[.!?…])\s+', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if not sentence.strip(): continue
            if len(current_chunk) + len(sentence) < max_chars:
                current_chunk += sentence + " "
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
            if len(combined) < fade_samples or len(next_chunk) < fade_samples:
                combined = np.concatenate((combined, next_chunk))
                continue
            fade_out = np.linspace(1, 0, fade_samples)
            fade_in = np.linspace(0, 1, fade_samples)
            tail = combined[-fade_samples:] * fade_out
            head = next_chunk[:fade_samples] * fade_in
            overlap = tail + head
            combined = np.concatenate((combined[:-fade_samples], overlap, next_chunk[fade_samples:]))
        return combined

    def _normalize_audio(self, audio_data, target_db=-1.0):
        max_val = np.abs(audio_data).max()
        if max_val == 0: return audio_data
        target_amp = 10 ** (target_db / 20)
        return audio_data * (target_amp / max_val)

    def process_narration(self, voice_key, raw_text, seed_base: int = 1234):
        """
        Main API method. Optimized for stability using presets.
        """
        if voice_key not in VOICE_PRESETS:
            logger.error(f"❌ Voice key '{voice_key}' not found.")
            return None, None

        # Load parameters from the preset
        params = VOICE_PRESETS[voice_key]

        # --- 1. Vocal DNA Caching (Load Reference) ---
        cache_key = (voice_key, params["ref_path"])
        if cache_key in self.vocal_dna_cache:
            audio_bytes = self.vocal_dna_cache[cache_key]
        else:
            logger.info(f"🧬 Encoding DNA for: {voice_key}")
            with open(params["ref_path"], "rb") as f:
                audio_bytes = f.read()
            self.vocal_dna_cache[cache_key] = audio_bytes

        # --- 2. Text Preparation ---
        # Clean and split text into manageable chunks (200 chars optimal)
        text_chunks = self.split_text(raw_text, max_chars=200)

        raw_audio_segments = []
        hist_tokens = None
        hist_text = None

        # Determine tags
        style_tags = params.get("style_tags", "")

        set_seed(seed_base)

        try:
            for i, chunk_text in enumerate(text_chunks):
                chunk_text = chunk_text.strip()
                if not chunk_text: continue

                logger.debug(f"⏳ Processing chunk {i + 1}/{len(text_chunks)}")

                # --- Strategy: Initial Tag Injection Only ---
                # Inject tags only on the first chunk to set the tone, then rely on context.
                # If you prefer constant injection, remove the 'if i == 0 else chunk_text' logic.
                processed_text = f"{style_tags} {chunk_text}" if (i == 0 and style_tags) else chunk_text

                # --- Auto-Retry Mechanism (The Judge) ---
                max_retries = 3
                best_attempt = None

                for attempt in range(max_retries):
                    # Slight seed variation for retries
                    if attempt > 0:
                        set_seed(seed_base + i + attempt * 100)

                    req = ServeTTSRequest(
                        text=processed_text,
                        references=[ServeReferenceAudio(audio=audio_bytes, text=params["prompt"])],
                        use_memory_cache="on",
                        chunk_length=params['chunk'],  # Use chunk size from preset (e.g., 300)
                        max_new_tokens=1024,  # Large buffer to prevent cuts
                        top_p=params['top_p'],
                        temperature=params['temp'],
                        repetition_penalty=params['penalty'],
                        format="wav",
                        prompt_text=[hist_text] if hist_text is not None else None,
                        prompt_tokens=[hist_tokens] if hist_tokens is not None else None,
                    )

                    final_res = None
                    for res in self.engine.inference(req):
                        if res.code == "final":
                            final_res = res
                            break

                    # --- Quality Check ---
                    if final_res and final_res.codes is not None:
                        num_tokens = final_res.codes.shape[1]

                        # Rule: Minimum 1 token per character (approx).
                        # Adjust based on language speed. Spanish usually ~1.2-1.4 tokens/char.
                        min_tokens_needed = len(chunk_text)

                        if num_tokens < min_tokens_needed:
                            logger.warning(f"⚠️ Chunk too short ({num_tokens} vs {len(chunk_text)} chars). Retrying...")
                            continue

                        best_attempt = final_res
                        break

                # If all retries fail, use the last result
                if best_attempt is None and final_res is not None:
                    logger.error(f"❌ Retries failed for chunk {i}. Using fallback.")
                    best_attempt = final_res

                if best_attempt is None or best_attempt.audio is None:
                    continue

                sr, audio_np = best_attempt.audio

                # Append audio segment
                raw_audio_segments.append(audio_np)

                # --- Context Update (Short Memory) ---
                # Keep only 50 tokens to maintain flow but prevent artifact accumulation (robotic voice)
                if best_attempt.codes is not None:
                    codes = torch.from_numpy(best_attempt.codes).to(torch.int)
                    keep = 50
                    if codes.shape[1] > keep:
                        codes = codes[:, -keep:]
                    hist_tokens = codes
                    hist_text = chunk_text

                # Clean VRAM
                torch.cuda.empty_cache()
                gc.collect()

            # --- 3. Post-Processing ---
            if raw_audio_segments:
                logger.info("🔧 Applying Crossfade and Normalization...")

                # Apply Crossfade (30ms for smoother transitions)
                merged = self._crossfade_chunks(raw_audio_segments, crossfade_ms=30)

                # Normalize
                final_audio = self._normalize_audio(merged)

                # Optional: Add silence padding at the end
                silence_pad = np.zeros(int(44100 * 0.5))
                final_audio = np.concatenate((final_audio, silence_pad))

                return final_audio, 44100

            return None, None

        except Exception as e:
            logger.error(f"🔥 Engine Error: {e}")
            import traceback
            traceback.print_exc()
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

    LONG_CHAPTER_2 = """
            Imagina por un momento que no eres simplemente un cuerpo físico luchando en el espacio, sino una frecuencia vibratoria, 
            una extensión directa de la inteligencia infinita... Nunca has estado separado de la totalidad... Esa sensación de soledad 
            es solo una ilusión óptica de la mente, un olvido temporal de tu verdadera naturaleza ilimitada y eterna que siempre 
            está conectada a la fuente.

            Entiende bien esto. El tiempo no es una línea recta hacia el futuro, es un vasto océano de posibilidades ocurriendo ahora mismo. 
            Tu deseo no está en un "mañana" lejano esperando ser alcanzado; está aquí, en una frecuencia paralela que aún no has 
            sintonizado. Al igual que una radio no crea la música, tú no "creas" tu realidad desde la nada, simplemente sintonizas 
            la versión de ti mismo que ya la está viviendo. La realidad física es solo el residuo de tus frecuencias pasadas.

            Si sigues observando lo que te falta, estás perpetuando la escasez. La realidad es arcilla fresca en manos de tu consciencia. 
            No puedes moldear una nueva figura si sigues aferrado a la forma antigua. Pregúntate: ¿Qué sentiría si mi deseo ya fuera un hecho? 
            El universo no entiende de súplicas, entiende de resonancia. Si vibras en "necesidad", atraerás más necesidad. 
            Si vibras en "gratitud", atraerás motivos infinitos para agradecer.

            Así pues, la maestría no reside en manipular el mundo externo, sino en conquistar tu diálogo interno. Se trata de 
            habitar el estado del deseo cumplido con tanta convicción que la evidencia física se vuelva irrelevante. Camina con 
            la certeza absoluta de quien ya posee el tesoro. Cuando esa paz inquebrantable se instala en tu pecho, el mundo físico 
            no tiene otra opción que ceder y moldearse a tu nueva frecuencia... Inevitablemente, te convertirás en lo que sientes que eres.
        """

    audio_data, sample_rate = engine.process_narration(
        voice_key="CAMILA",
        raw_text=LONG_CHAPTER_2
    )

    if audio_data is not None:
        output_path = "output_camila_optimized.wav"
        sf.write(output_path, audio_data, sample_rate, subtype="PCM_16")
        logger.success(f"🏆 Audio generated successfully: {output_path}")
    else:
        logger.error("❌ Audio generation failed.")