import os
import platform
import sys

# --- ENVIRONMENT AGNOSTIC ARCHITECTURE ---
# Detect platform before loading heavy ML libraries
IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    # Performance optimization strictly for Windows 4GB VRAM.
    # CRITICAL: MUST be set before importing torch.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import io
import re
import gc
import shutil
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from loguru import logger
from datetime import datetime


# --- SYSTEM CONFIGURATION ---
logger.remove()
logger.add(sys.stdout, colorize=True, level="TRACE",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

# --- Constants for Directory Paths ---
# Auto-detect project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- IMPORTS WITH FALLBACK ---
try:
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.dac.inference import load_model as load_decoder_model
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
    from fish_speech.utils import set_seed
except ImportError:
    # If running from a notebook cell, add root to path
    sys.path.insert(0, str(PROJECT_ROOT))
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.dac.inference import load_model as load_decoder_model
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
    from fish_speech.utils import set_seed

# --- VOICE PRESETS (OPTIMIZED FOR S1-MINI) ---
# NOTE: Temperatures lowered to ~0.75 and Penalty to ~1.05 to prevent metallic artifacts.
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
        "temp": 0.70,  # FIXED: Lowered from 0.88 to prevent "shrill" voice
        "top_p": 0.70,  # Tighter control to avoid robotic drifting
        "chunk": 300,  # Safe size for Quadro T1000
        "penalty": 1.035,  # Increased to prevent loop/stuttering
        "ref_path": str(PROJECT_ROOT / "voices" / "Camila_Sodi.mp3"),
        "prompt": """Todos venimos de un mismo campo fuente, de una misma gran energía, de un mismo Dios, de un mismo 
        universo, como le quieras llamar. Todos somos parte de eso. Nacemos y nos convertimos en esto por un ratito 
        muy chiquito, muy chiquitito, que creemos que es muy largo y se nos olvida que vamos a regresar a ese lugar 
        de donde venimos, que es lo que tú creas, adonde tú creas, pero inevitablemente vas a regresar.""",
        "style_tags": "(calm)(narrator)" #(deep voice)
    },
    "CRISTY": {
        "temp": 0.70,  # FIXED: Lowered from 0.88 to prevent "shrill" voice
        "top_p": 0.70,  # Tighter control to avoid robotic drifting
        "chunk": 300,  # Safe size for Quadro T1000
        "penalty": 1.035,  # Increased to prevent loop/stuttering
        "ref_path": str(PROJECT_ROOT / "voices" / "cristy_de_la_torre.wav"),
        "prompt": """Todo es mente, el universo es mental y Jesús lo dijo sin rodeos, Jesús decía si tuvieras fe como 
        un grano de mostaza, le dirías a la montaña, muévete y la montaña se movería. Esto no es metáfora poética, 
        esto es ingeniería de conciencia, la fe no es creer algo bonito, la fe es alinear tu mente con la mente 
        divina, con la mente universal, por eso Jesús buscaba la soledad, porque los milagros no se dan en el ruido, 
        se dan en el silencio, la preocupación es caos mental, la fe es coherencia. Pregúntate esto sin huir, 
        ¿qué pienso todos los días? ¿qué mundo está creando mi mente? Jesús y Hermes lo sabían, el mundo responde a 
        la calidad de tu atención, por eso donde pones tu atención pones tu energía.""" ,
        # S1-mini tag strategy for Spanish:
        # Only (soft tone) — single tone marker, most stable in S1-mini Spanish.
        # (narrator) caused volume spikes and occasional vocalization — removed.
        # (soft tone) alone keeps the voice grounded without triggering artifacts.
        "style_tags": "(soft tone)",
        "paragraph_tags": [
            "(soft tone)",   # P1 Hook
            "(soft tone)",   # P2 Truth
            "(soft tone)",   # P3 Resolution
            "(soft tone)",   # P4 CTA
        ]
    },
    "ADAM": {
        "temp": 0.70,
        "top_p": 0.70,
        "chunk": 300,
        "penalty": 1.15,
        "ref_path": str(PROJECT_ROOT / "voices" / "adam_spanish.wav"),
        "prompt": """La noche ya no es un tiempo perdido para la energía. Aunque el Sol se oculte, la ciencia ha 
        demostrado que todavía es posible obtener significativas cantidades de electricidad cuando todo parece 
        apagado. Parte de este avance consiste en aprovechar la luz de la Luna, que es luz solar muy débil, 
        pero suficiente para alimentar sensores y dispositivos de bajo consumo. No sirve para grandes sistemas, 
        pero sí para mantener una energía mínima activa durante la noche. Las esferas, como la de la imagen, 
        están hechas de cristal industrial de 45 mm de espesor y pueden instalarse solas o en conjuntos. Además, 
        pueden integrarse en sistemas de seguimiento automático de fuentes de luz, como el Sol o la Luna, incluso en 
        condiciones de baja iluminación.""",
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

        # Use half precision (FP16) universally for speed and memory efficiency
        self.precision = torch.float16

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
        text = re.sub(r'([.!?…])(?=\S)', r'\1 ', text)
        text = text.replace("\n", " ").replace("\t", " ")
        return re.sub(r'\s+', ' ', text).strip()

    def split_text(self, text, max_chars=200):
        """
        HYBRID SPLIT STRATEGY v2 — PARAGRAPH-FIRST + FATIGUE CONTROL:

        Priority order:
        1. Split by double newlines (\\n\\n) — paragraph boundaries = natural pauses.
           This MUST happen BEFORE clean_text, which strips all newlines.
        2. If a paragraph exceeds max_chars, split internally by sentence.

        Why paragraph-first matters:
        The script from the LLM arrives with 4 paragraphs (Hook / Truth / Resolution / CTA).
        Each paragraph has its own emotional weight and pacing.
        Flattening them into 1 chunk loses all natural pause structure.
        Preserving \\n\\n as chunk boundaries = natural pause between paragraphs.

        Args:
            text (str): Input text. May contain \\n\\n paragraph breaks.
            max_chars (int): Safety limit per chunk. If a paragraph exceeds this,
                             it gets split by sentence. Recommended: 200-300.
        """
        if not text:
            return []

        chunks = []

        # Step 1: Split by paragraph breaks FIRST (before clean_text strips them)
        # \n\s*\n matches blank lines even when they contain spaces/tabs (e.g. indented test strings)
        paragraphs = re.split(r'\n\s*\n', text.strip())

        for paragraph in paragraphs:
            # Clean each paragraph individually (preserves cross-paragraph boundaries)
            clean_para = self.clean_text(paragraph)
            if not clean_para:
                continue

            if len(clean_para) <= max_chars:
                # Paragraph fits in one chunk — keep it whole
                chunks.append(clean_para)
            else:
                # Paragraph too long — split by sentence (fatigue control)
                sentences = re.split(r'(?<=[.!?…])\s+', clean_para)
                current_chunk = ""
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    if len(current_chunk) + len(sentence) < max_chars:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
                if current_chunk:
                    chunks.append(current_chunk.strip())

        return chunks

    def _crossfade_chunks(self, audio_list, crossfade_ms=30, sample_rate=44100):
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

    def _load_and_trim_audio(self, file_path, max_duration=60):
        """
        Carga y recorta el audio automáticamente si es muy largo para salvar la VRAM.
        """
        try:
            data, sr = sf.read(file_path)

            if len(data) > sr * max_duration:
                logger.warning(
                    f"✂️ Audio too long ({len(data) / sr:.1f}s). Trimming to {max_duration}s to prevent OOM.")
                data = data[:int(sr * max_duration)]

            buffer = io.BytesIO()
            sf.write(buffer, data, sr, format='WAV')
            return buffer.getvalue()

        except Exception as e:
            logger.error(f"Error loading audio {file_path}: {e}")
            with open(file_path, "rb") as f:
                return f.read()

    def process_narration(self, voice_key, raw_text, seed_base: int = 1234):
        """
        Main API method. Optimized for stability using presets.
        """
        if voice_key not in VOICE_PRESETS:
            logger.error(f"❌ Voice key '{voice_key}' not found.")
            return None, None

        # Load parameters from the preset
        params = VOICE_PRESETS[voice_key]
        set_seed(seed_base)

        # --- 1. Vocal DNA Caching (Load Reference) ---
        cache_key = (voice_key, params["ref_path"])
        if cache_key in self.vocal_dna_cache:
            audio_bytes = self.vocal_dna_cache[cache_key]
        else:
            # ENVIRONMENT AGNOSTIC MAGIC:
            # 12 seconds for constrained Windows laptops, 60 seconds for Cloud GPUs
            safe_duration = 12 if IS_WINDOWS else 60
            audio_bytes = self._load_and_trim_audio(params["ref_path"], max_duration=safe_duration)
            self.vocal_dna_cache[cache_key] = audio_bytes

        # --- 2. Text Preparation ---
        # Clean and split text into manageable chunks (200 chars optimal)
        text_chunks = self.split_text(raw_text, max_chars=200)

        raw_audio_segments = []
        hist_tokens = None
        hist_text = None

        # Determine tags
        current_tags = params.get("style_tags", "")
        paragraph_tags = params.get("paragraph_tags", [])

        try:
            for i, chunk_text in enumerate(text_chunks):
                chunk_text = chunk_text.strip()
                if not chunk_text: continue

                logger.debug(f"⏳ Processing chunk {i + 1}/{len(text_chunks)}")

                # --- Tag Injection Strategy ---
                # Official docs: emotion tags MUST go at the beginning of sentences.
                #
                # CRITICAL: If the chunk already starts with a marker pattern like
                # "(concerned)", "(empathetic)", etc. — the text was pre-tagged by the
                # LLM or the test script. Do NOT inject paragraph_tags on top.
                # Injecting twice causes the model to read the tag aloud as text.
                #
                # Detection: a chunk is pre-tagged if it starts with "(<word>)"
                _already_tagged = bool(re.match(r'^\([a-z\s\-]+\)', chunk_text.strip()))

                if _already_tagged:
                    # Text has its own markers — pass through untouched
                    processed_text = chunk_text
                elif paragraph_tags:
                    # Engine injects per-paragraph tags (cyclically P1→P2→P3→P4)
                    tag = paragraph_tags[i % len(paragraph_tags)]
                    processed_text = f"{tag} {chunk_text}"
                elif current_tags:
                    # Inject same style_tags on every chunk (prevents robotic drift)
                    processed_text = f"{current_tags} {chunk_text}"
                else:
                    processed_text = chunk_text

                # --- Auto-Retry Mechanism (The Judge) ---
                max_retries = 3
                best_attempt = None

                for attempt in range(max_retries):
                    # Slight seed variation for retries
                    if attempt > 0:
                        set_seed(seed_base + i + attempt * 100)

                    # Full ServeTTSRequest with all parameters
                    # (previously commented out — re-enabled for quality control)
                    req = ServeTTSRequest(
                        text=processed_text,
                        references=[ServeReferenceAudio(audio=audio_bytes, text=params["prompt"])],
                        use_memory_cache="on",
                        chunk_length=params['chunk'],        # Chunk size from preset (300)
                        max_new_tokens=1024,                 # Large buffer to prevent cuts
                        top_p=params['top_p'],               # Sampling control
                        temperature=params['temp'],          # Voice variability
                        repetition_penalty=params['penalty'],# Prevents loops/stuttering
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
                padding_samples = int(sr * 0.25)
                silence_pad = np.zeros(padding_samples, dtype=audio_np.dtype)
                audio_padded = np.concatenate((audio_np, silence_pad))
                raw_audio_segments.append(audio_padded)

                # --- Context Update (Short Memory) ---
                # Keep only 50 tokens to maintain flow but prevent artifact accumulation.
                # Store processed_text (WITH tags) so the next chunk's context
                # includes the emotional instruction, not just the bare text.
                if best_attempt.codes is not None:
                    codes = torch.from_numpy(best_attempt.codes).to(torch.int)
                    # keep = 50
                    # if codes.shape[1] > keep:
                    #     codes = codes[:, -keep:]
                    hist_tokens = codes
                    hist_text = processed_text  # WITH tags — preserves emotional context

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

    def apply_custom_lora(self, lora_path):
        """
        Reemplaza el motor Llama actual con una versión que usa los pesos entrenados (LoRA).
        Mantiene el decodificador original (codec.pth) intacto.
        """
        lora_path = Path(lora_path)
        if not lora_path.exists():
            logger.error(f"❌ No se encontró el checkpoint LoRA en: {lora_path}")
            return False

        logger.info(f"🔄 Cargando Pesos Entrenados (LoRA): {lora_path.name}")

        try:
            # Re-lanzamos la cola de inferencia apuntando al archivo .ckpt
            # Fish Speech loads the base model and applies the LoRA patch automatically.
            new_llama_queue = launch_thread_safe_queue(
                checkpoint_path=lora_path,
                device=self.device,
                precision=self.precision,
                compile=self.should_compile
            )

            # Actualizamos la cola en el motor existente
            self.engine.llama_queue = new_llama_queue

            logger.success(f"✅ LoRA '{lora_path.name}' aplicado con éxito al motor.")
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            logger.error(f"❌ Error al aplicar LoRA: {e}")
            return False

    def load_latest_camila_checkpoint(self, results_dir=PROJECT_ROOT / "checkpoints" / "camila_voice_v1_stable"):
        """
        Función de conveniencia para buscar y aplicar el checkpoint más nuevo de Camila.
        """

        checkpoints_path = Path(results_dir)
        if not checkpoints_path.exists():
            logger.error("❌ No existe la carpeta de checkpoints de Camila.")
            return False

        # Find all .ckpt files and pick the most recently modified one
        list_of_files = list(checkpoints_path.glob("*.ckpt"))
        if not list_of_files:
            logger.warning("⚠️ No se encontraron archivos .ckpt todavía.")
            return False

        latest_file = max(list_of_files, key=os.path.getmtime)
        return self.apply_custom_lora(latest_file)

    def apply_lora(self, checkpoint_path):
        """
        Carga los pesos entrenados (.ckpt) en el motor actual.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            logger.error(f"❌ No se encontró el checkpoint en: {path}")
            return False

        logger.info(f"🔄 Aplicando conocimiento entrenado (LoRA): {path.name}")
        try:
            # Re-lanzamos la cola de inferencia con el nuevo checkpoint
            self.engine.llama_queue = launch_thread_safe_queue(
                checkpoint_path=path,
                device=self.device,
                precision=self.precision,
                compile=self.should_compile
            )
            logger.success("✅ Pesos LoRA cargados correctamente.")
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            logger.error(f"❌ Error al cargar LoRA: {e}")
            return False

    def speak_camila(self, text, seed=42):
        """
        FUNCIÓN SIMPLIFICADA:
        Solo pasas el texto. Usa automáticamente el preset de Camila y el
        LoRA que esté cargado en el motor.
        """
        logger.info(f"🎙️ Generando voz entrenada de Camila...")
        # Reuse the narration logic but lock the preset to CAMILA
        return self.process_narration(voice_key="CAMILA", raw_text=text, seed_base=seed)


# --- TESTING BLOCK ---
if __name__ == "__main__":
    engine = FishTTSEngine()

    # TEXTO DE PRUEBA
    LONG_CHAPTER = """
            Todos venimos de un mismo campo fuente, de una misma gran energía, de un mismo Dios, de un mismo
            universo, como le quieras llamar. Todos somos parte de eso. Nacemos y nos convertimos en esto por un ratito,
            muy chiquito, muy chiquitito, que creemos que es muy largo y se nos olvida que vamos a regresar a ese lugar
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
            clic, el mundo exterior no tiene más remedio que reorganizarse para reflejar tu nueva verdad. E inevitablemente,
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


    # ===========================================================================
    # TEST A — Dark Stoic script with per-paragraph emotion markers
    # Structure: 4 paragraphs separated by \n\n
    # Engine processes each paragraph as an independent chunk with its own tag
    # Markers per Fish Audio S1 official docs:
    #   (concerned)   → P1 Hook: names the symptom with care
    #   (empathetic)  → P2 Uncomfortable truth: understands from lived experience
    #   (comforting)  → P3 Resolution: grounded, not preachy
    #   (sincere)     → P4 CTA: direct invitation
    # Paralanguage (fine-grained control, V1.6):
    #   (break)       → short pause between ideas
    #   (breath)      → natural breath before a hard truth
    #   (sigh)        → natural sigh at emotional weight moments
    # ===========================================================================
    DARK_STOIC_TEST = """Llevas años sintiéndote exhausta. Y ya ni recuerdas cuándo no fue así. Eso que sientes al despertar no es tu edad. Es tu cuerpo diciéndote que algo no está bien.

        No es el trabajo. Es que llevas años cargando lo que no te corresponde. Tu sistema nervioso aprendió que descansar era peligroso, que si parabas, algo se rompía. Y así llevas, sosteniéndolo todo, para todos, menos para ti.
        
        Eso puede soltarse. No de golpe, pero puede. No tienes que seguir cargando lo que no te pidieron que cargaras. Hay una salida que no implica seguir aguantando.
        
        Comenta la palabra CALMA y te mando el protocolo.
    """

    # ===========================================================================
    # TEST B — Short text to quickly validate the pipeline
    # ===========================================================================
    QUICK_TEST = """Tu cansancio no es normal.

        Llevas años siendo suficiente para todos menos para ti.
        
        Comenta CALMA si esto te llegó.
    """

    # Switch between DARK_STOIC_TEST (full) and QUICK_TEST (fast validation)
    TEST_TEXT = DARK_STOIC_TEST

    logger.info(f"🧪 Test text chunks preview:")
    preview_chunks = engine.split_text(TEST_TEXT, max_chars=200)
    for idx, chunk in enumerate(preview_chunks):
        logger.info(f"  Chunk {idx + 1}: {chunk[:80]}...")

    audio_data, sample_rate = engine.process_narration(
        voice_key="CRISTY",
        raw_text=TEST_TEXT
    )

    if audio_data is not None:
        output_path = "output_dark_stoic_test.wav"
        sf.write(output_path, audio_data, sample_rate, subtype="PCM_16")
        logger.success(f"🏆 Audio generated: {output_path}")
        logger.info(f"⏱️  Duration: {len(audio_data) / sample_rate:.2f}s")
    else:
        logger.error("❌ Audio generation failed.")

    # # 2. CARGAR TU ENTRENAMIENTO (El archivo .ckpt que bajaste)
    # # Reemplaza con la ruta real de tu archivo
    # checkpoint_camila = "/workspace/fish-speech/results/camila_voice_v1_stable/checkpoints/step_000000500.ckpt"
    #
    # if engine.apply_lora(checkpoint_camila):
    #
    #     # 3. GENERACIÓN SIMPLIFICADA (Solo pasas el texto)
    #     texto_a_decir = """
    #             Hola, soy Camila. Esta es una prueba usando mi entrenamiento personalizado.
    #             Como puedes ver, ahora el código es mucho más limpio. Solo me pasas el texto
    #             y yo me encargo de sonar exactamente como tú esperas.
    #         """
    #
    #     audio, sr = engine.speak_camila(texto_a_decir)
    #
    #     # 4. Guardar el resultado
    #     if audio is not None:
    #         output_name = "camila_trained_simple_test.wav"
    #         sf.write(output_name, audio, sr)
    #         logger.success(f"🏆 ¡Listo! Audio generado en: {output_name}")
    # else:
    #     logger.error("No se pudo aplicar el entrenamiento, revisa la ruta del .ckpt")
