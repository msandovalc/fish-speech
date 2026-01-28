import io
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

# --- CONSTANTS ---
# Auto-detect project root
PROJECT_ROOT = Path(__file__).resolve().parent

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
    # "MARLENE": {
    #     "temp": 0.65,
    #     "top_p": 0.70,
    #     "chunk": 300,
    #     "penalty": 1.035,
    #     "ref_path": str(PROJECT_ROOT / "voices" / "ElevenLabs_Marlene.mp3"),
    #     "prompt": """La mente lo es todo. La causa mental. La causa de todo -absolutamente todo- es mental, es decir,
    #     la mente es la que produce o causa todo en la vida del individuo.
    #
    #     Cuando reconozcamos, entendamos y aceptemos esta verdad, habremos dado un paso muy importante en el progreso del desarrollo.
    #
    #     Si todo es mental, este es un universo mental, donde todo funciona por medios mentales. Nosotros somos seres
    #     mentales, mentalidades buenas, perfectas y eternas.
    #
    #     La mente sólo tiene una actividad, pensar. El pensamiento es todo lo de la mente lo único que somos y tenemos es
    #     pensamiento, por ello, el pensamiento es lo más importante de todo.
    #     """,
    #     "style_tags": "(calm) (narrator)"
    # },
    # "MARGARITA": {
    #     "temp": 0.65,
    #     "top_p": 0.70,
    #     "chunk": 300,
    #     "penalty": 1.035,
    #     "ref_path": str(PROJECT_ROOT / "voices" / "Margarita_Navarrete.wav"),
    #     "prompt": """Mira te comparto, hicimos tres cuartos más y no suelta todavía el sistema y otros detallitos,
    #     pero mira lo que te quiero comentar es que sé que suena raro, sé que se requiere dinero para el intercambio
    #     de lo que se desea, sin embargo todo lo que decidas hacer, hazlo porque deseas hacerlo. Lo común es buscar
    #     hacerlo porque necesitas, y entonces si se empieza a hacer todo desde la necesidad, desde pues es que Magui
    #     si lo requiero para los pagos, quedó bien justito ahorita, entonces te me vas a empezar a estresar más. Haz
    #     las cosas porque te gusta lo que estás haciendo y de lo que te gusta empieza a hacer más, pero porque te gusta.
    #
    #     ¿Cómo voy a poder eliminar la carencia del gusto? Por eso son las líneas, a mí me pasó, te digo tiene poco
    #     que saque el crédito.
    #     """,
    #     "style_tags": "(calm) (narrator)"
    # },
    "CAMILA": {
        "temp": 0.65,
        "top_p": 0.70,
        "chunk": 300,
        "penalty": 1.035, #1.035
        "ref_path": str(PROJECT_ROOT / "voices" / "cami_sodi_50_secs.mp3"),
        "prompt": """Así  el  niño  se  engaña  fácilmente,  toma  mentiras  o  falsedades  como verdades, 
        solo porque las ve u oye, se engaña como un niño. El niño, por ejemplo, ve que el sol sale por el oriente, 
        asciende en el firmamento; está en el centro o cenit al mediodía y continuará su camino hacia el poniente, 
        donde se pone u oculta, así, el sol realiza dicho recorrido todos los días, y para el niño que ve eso, 
        es la verdad; pero, si tuviera la base de la realidad para razonar, de que el sol, centro del sistema solar, 
        no se mueve en ese sentido, sino que es la tierra la que se mueve, aunque la apariencia sea de que es el sol 
        el que se mueve, entonces el niño sabría que el mencionado movimiento del sol es una ilusión. """,
        "style_tags": "(calm)(narrator)(deep voice)" #(deep voice)
    }
    # "ROSA": {
    #     "temp": 0.65,
    #     "top_p": 0.70,
    #     "chunk": 300,
    #     "penalty": 1.035,
    #     "ref_path": str(PROJECT_ROOT / "voices" / "Elevenlabs_Rosa_Estela.wav"),
    #     "prompt": """El agua, la confianza y el miedo. Una lección poderosa y reveladora sobre la verdadera
    #     protección y el poder de la preparación. Considera la profunda enseñanza que subyace a la instrucción sobre
    #     el miedo al agua.
    #
    #     Inbuir en la mente de un niño pequeño un temor paralizante hacia la profundidad, creyendo erróneamente que
    #     así se le protege de un posible ahogamiento, puede paradójicamente paralizarlo por completo en un momento de
    #     peligro real, impidiéndole reaccionar de manera efectiva para salvar su propia vida. En contraste,
    #     enseñar al niño un amor genuino por el agua como una parte esencial y maravillosa de la naturaleza,
    #     inculcarle un respeto saludable por su poder y, lo que es crucial, dotarlo de la habilidad vital de nadar con
    #     confianza, empodera al niño de una manera transformadora. Esta analogía poderosa se extiende a innumerables
    #     otros temores que, con las mejores intenciones pero con resultados a menudo limitantes, se nos transmiten
    #     desde la infancia.
    #
    #     ¿Cuáles son esas aguas profundas metafóricas que has estado evitando en tu vida por un temor arraigado,
    #     impidiéndote explorar nuevas oportunidades y experiencias enriquecedoras? Comparte tu profunda reflexión en
    #     los comentarios. Dale like a este video si crees firmemente en el poder de la preparación activa y la
    #     confianza cultivada como la verdadera protección contra los desafíos de la vida, en lugar de la evitación
    #     basada en el miedo, y sígueme para explorar juntos más analogías reveladoras que iluminan la naturaleza del
    #     temor y el camino hacia la liberación.""" ,
    #     "style_tags": "(calm) (narrator) (relaxed)" #(deep voice)
    # }
    # "ALEJANDRO": {
    #     "temp": 0.65,
    #     "top_p": 0.70,
    #     "chunk": 300,
    #     "penalty": 1.035,
    #     "ref_path": str(PROJECT_ROOT / "voices" / "ElevenLabs_Alejandro.mp3"),
    #     "prompt": """(serious) (calm) La mente lo es todo. La causa mental. La causa de todo -absolutamente todo- es mental, es decir,
    #         la mente es la que produce o causa todo en la vida del individuo.
    #         Cuando reconozcamos, entendamos y aceptemos esta verdad, habremos dado un paso muy importante en el progreso del desarrollo.
    #         Si todo es mental, este es un universo mental, donde todo funciona por medios mentales. Nosotros somos seres
    #         mentales, mentalidades buenas, perfectas y eternas.
    #         La mente sólo tiene una actividad, pensar. El pensamiento es todo lo de la mente lo único que somos y tenemos es
    #         pensamiento, por ello, el pensamiento es lo más importante de todo.""",
    #     "style_tags": "(calm) (narrator)"
    # },
    # "ALEJANDRO_BALLESTEROS": {
    #     "temp": 0.65,
    #     "top_p": 0.70,
    #     "chunk": 300,
    #     "penalty": 1.035,
    #     "ref_path": str(PROJECT_ROOT / "voices" / "Elevenlabs_Alejandro_Ballesteros.wav"),
    #     "prompt": """El agua, la confianza y el miedo. Una lección poderosa y reveladora sobre la verdadera
    #     protección y el poder de la preparación. Considera la profunda enseñanza que subyace a la instrucción sobre
    #     el miedo al agua.
    #
    #     Inbuir en la mente de un niño pequeño un temor paralizante hacia la profundidad, creyendo erróneamente que
    #     así se le protege de un posible ahogamiento, puede paradójicamente paralizarlo por completo en un momento de
    #     peligro real, impidiéndole reaccionar de manera efectiva para salvar su propia vida. En contraste,
    #     enseñar al niño un amor genuino por el agua como una parte esencial y maravillosa de la naturaleza,
    #     inculcarle un respeto saludable por su poder y, lo que es crucial, dotarlo de la habilidad vital de nadar con
    #     confianza, empodera al niño de una manera transformadora. Esta analogía poderosa se extiende a innumerables
    #     otros temores que, con las mejores intenciones pero con resultados a menudo limitantes, se nos transmiten
    #     desde la infancia.
    #
    #     ¿Cuáles son esas aguas profundas metafóricas que has estado evitando en tu vida por un temor arraigado,
    #     impidiéndote explorar nuevas oportunidades y experiencias enriquecedoras? Comparte tu profunda reflexión en
    #     los comentarios. Dale like a este video si crees firmemente en el poder de la preparación activa y la
    #     confianza cultivada como la verdadera protección contra los desafíos de la vida, en lugar de la evitación
    #     basada en el miedo, y sígueme para explorar juntos más analogías reveladoras que iluminan la naturaleza del
    #     temor y el camino hacia la liberación.""",
    #     "style_tags": "(calm) (narrator)"
    # },
    # "ENRIQUE": {
    #     "temp": 0.65,
    #     "top_p": 0.70,
    #     "chunk": 300,
    #     "penalty": 1.035,
    #     "ref_path": str(PROJECT_ROOT / "voices" / "Elevenlabs_Enrique_Nieto.wav"),
    #     "prompt": """El agua, la confianza y el miedo. Una lección poderosa y reveladora sobre la verdadera
    #     protección y el poder de la preparación. Considera la profunda enseñanza que subyace a la instrucción sobre
    #     el miedo al agua.
    #
    #     Inbuir en la mente de un niño pequeño un temor paralizante hacia la profundidad, creyendo erróneamente que
    #     así se le protege de un posible ahogamiento, puede paradójicamente paralizarlo por completo en un momento de
    #     peligro real, impidiéndole reaccionar de manera efectiva para salvar su propia vida. En contraste,
    #     enseñar al niño un amor genuino por el agua como una parte esencial y maravillosa de la naturaleza,
    #     inculcarle un respeto saludable por su poder y, lo que es crucial, dotarlo de la habilidad vital de nadar con
    #     confianza, empodera al niño de una manera transformadora. Esta analogía poderosa se extiende a innumerables
    #     otros temores que, con las mejores intenciones pero con resultados a menudo limitantes, se nos transmiten
    #     desde la infancia.
    #
    #     ¿Cuáles son esas aguas profundas metafóricas que has estado evitando en tu vida por un temor arraigado,
    #     impidiéndote explorar nuevas oportunidades y experiencias enriquecedoras? Comparte tu profunda reflexión en
    #     los comentarios. Dale like a este video si crees firmemente en el poder de la preparación activa y la
    #     confianza cultivada como la verdadera protección contra los desafíos de la vida, en lugar de la evitación
    #     basada en el miedo, y sígueme para explorar juntos más analogías reveladoras que iluminan la naturaleza del
    #     temor y el camino hacia la liberación.""",
    #     "style_tags": "(calm) (narrator)"
    # }
}

# Platform detection
is_windows = platform.system() == "Windows"
should_compile = False if is_windows else True


class FishTotalLab:
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
        text = re.sub(r'([.!?…])(?=\S)', r'\1 ', text)
        text = text.replace("\n", " ").replace("\t", " ")
        return re.sub(r'\s+', ' ', text).strip()

    def split_text(self, text, max_chars=200):
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

    def generate_audio_for_params(self, voice_key, raw_text, temp, top_p, penalty, chunk_size, style_tags, seed_base: int = 1234):
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
            audio_bytes = self._load_and_trim_audio(params["ref_path"], max_duration=60)
            self.vocal_dna_cache[cache_key] = audio_bytes

        # --- 2. Text Preparation ---
        # Clean and split text into manageable chunks (200 chars optimal)
        text_chunks = self.split_text(raw_text, max_chars=200)

        raw_audio_segments = []
        hist_tokens = None
        hist_text = None

        # Determine tags
        current_tags = style_tags if style_tags else params.get("style_tags", "")


        try:
            for i, chunk_text in enumerate(text_chunks):
                chunk_text = chunk_text.strip()
                if not chunk_text: continue

                logger.debug(f"⏳ Processing chunk {i + 1}/{len(text_chunks)}")

                # --- Strategy: Initial Tag Injection Only ---
                # Inject tags only on the first chunk to set the tone, then rely on context.
                # If you prefer constant injection, remove the 'if i == 0 else chunk_text' logic.
                processed_text = f"{current_tags} {chunk_text}" if (i == 0 and current_tags) else chunk_text
                #processed_text = f"{chunk_text}"

                # --- Auto-Retry Mechanism (The Judge) ---
                max_retries = 3
                best_attempt = None

                for attempt in range(max_retries):
                    # Slight seed variation for retries
                    if attempt > 0:
                        set_seed(seed_base + i + attempt * 100)

                    req = ServeTTSRequest(
                        text=processed_text,
                        references=[ServeReferenceAudio(audio=audio_bytes,
                                                        text=params["prompt"]
                                                        )],
                        use_memory_cache="on",
                        chunk_length=params['chunk'],  # Use chunk size from preset (e.g., 300)
                        max_new_tokens=1024,  # Large buffer to prevent cuts
                        top_p=params['top_p'],
                        temperature=params['temp'],
                        repetition_penalty=params['penalty'],
                        format="wav",
                        prompt_text=[hist_text] if hist_text is not None else None,
                        prompt_tokens=[hist_tokens] if hist_tokens is not None else None
                    )

                    # req = ServeTTSRequest(
                    #     text=processed_text,
                    #     references=[ServeReferenceAudio(
                    #         audio=audio_bytes,
                    #         text=params["prompt"]
                    #     )],
                    #     # Opcional: Si quieres que recuerde cachés anteriores para ir más rápido, déjalo "on".
                    #     # El default es "off", pero "on" no afecta la calidad, solo la velocidad.
                    #     use_memory_cache="on",
                    #
                    #     # IMPORTANTE: Si estás pasando historial (contexto previo), déjalo.
                    #     # Si quieres una prueba 100% limpia desde cero, borra estas dos líneas también.
                    #     # prompt_text=[hist_text] if hist_text is not None else None,
                    #     # prompt_tokens=[hist_tokens] if hist_tokens is not None else None
                    # )

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

    def run_hyper_search(self, text, num_tests=5):
        logger.info(f"🧪 Starting Hyper-Search for {len(VOICE_PRESETS)} voices.")
        timestamp = datetime.now().strftime("%H%M%S")

        for voice_name, base_params in VOICE_PRESETS.items():
            voice_folder = PROJECT_ROOT / f"LAB_{voice_name}_{timestamp}"
            voice_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"🔬 Testing Voice: {voice_name}")

            for i in range(num_tests):
                curr_temp = base_params['temp']
                curr_pen = base_params['penalty']
                curr_chunk = base_params['chunk']

                logger.trace(f"🌀 Test {i + 1}: Chunk Size={curr_chunk} | (T={curr_temp}, P={curr_pen})")

                result_tuple = self.generate_audio_for_params(
                    voice_name,
                    text,
                    temp=curr_temp,
                    top_p=base_params['top_p'],
                    penalty=curr_pen,
                    chunk_size=curr_chunk,
                    style_tags=base_params.get("style_tags", "")
                )

                if result_tuple is not None and result_tuple[0] is not None:
                    audio, sample_rate = result_tuple
                    filename = f"{voice_name}_FinalFixed_{timestamp}.wav"
                    sf.write(str(voice_folder / filename), audio, sample_rate, subtype="PCM_16")
                    logger.success(f"📦 Audio Successful Generated: {filename}")

            shutil.make_archive(str(PROJECT_ROOT / f"RESULTS_{voice_name}_{timestamp}"), 'zip', voice_folder)
            logger.success(f"📦 Test pack created for {voice_name}")

    # def run_hyper_search(self, text, num_tests=1):
    #     """
    #     LABORATORIO MATRICIAL:
    #     Itera sobre Rango de Temperaturas x Variaciones de Tags.
    #     """
    #     logger.info(f"🧪 Starting Hyper-Search for {len(VOICE_PRESETS)} voices.")
    #     timestamp = datetime.now().strftime("%H%M%S")
    #
    #     # --- 🎛️ CONFIGURACIÓN DEL LABORATORIO ---
    #     # 1. Barrido de Temperaturas (Estabilidad vs Creatividad)
    #     test_temps = [0.65, 0.66, 0.67, 0.68, 0.69, 0.70]
    #
    #     # 2. Parámetros Fijos (Ganadores)
    #     fixed_top_p = 0.70
    #     fixed_penalty = 1.035
    #
    #     # 3. Variaciones de Tags a probar por cada temperatura
    #     tag_variations = [
    #         "(calm)",
    #         "(calm) (narrator)",
    #         "(narrator)",
    #         "(calm) (narrator) (deep voice)"
    #     ]
    #     # ------------------------------------------
    #
    #     for voice_name, base_params in VOICE_PRESETS.items():
    #         if voice_name != "CAMILA": continue
    #
    #         voice_folder = PROJECT_ROOT / f"LAB_{voice_name}_{timestamp}"
    #         voice_folder.mkdir(parents=True, exist_ok=True)
    #         logger.info(f"🔬 Testing Voice: {voice_name}")
    #
    #         # Bucle 1: Temperaturas
    #         for curr_temp in test_temps:
    #
    #             # Bucle 2: Variaciones de Tags
    #             for i, current_tags in enumerate(tag_variations):
    #                 curr_chunk = base_params['chunk']
    #
    #                 # Crear nombre limpio para el archivo (ej: calm_narrator)
    #                 tag_suffix = current_tags.replace("(", "").replace(")", "").replace(" ", "_").strip("_")
    #
    #                 logger.trace(f"🌀 Test T={curr_temp} | Tags='{current_tags}'")
    #
    #                 result_tuple = self.generate_audio_for_params(
    #                     voice_name,
    #                     text,
    #                     temp=curr_temp,
    #                     top_p=fixed_top_p,
    #                     penalty=fixed_penalty,
    #                     chunk_size=curr_chunk,
    #                     style_tags=current_tags,
    #                     seed_base=1234 + i  # Variar semilla ligeramente por cada tag
    #                 )
    #
    #                 if result_tuple is not None and result_tuple[0] is not None:
    #                     audio, sample_rate = result_tuple
    #
    #                     # Nombre descriptivo: CAMILA_T0.65_calm_narrator.wav
    #                     filename = f"{voice_name}_T{curr_temp}_{tag_suffix}_{timestamp}.wav"
    #
    #                     sf.write(str(voice_folder / filename), audio, sample_rate, subtype="PCM_16")
    #                     logger.success(f"📦 Generated: {filename}")
    #
    #         shutil.make_archive(str(PROJECT_ROOT / f"RESULTS_{voice_name}_{timestamp}"), 'zip', voice_folder)
    #         logger.success(f"📦 ZIP ready for {voice_name}")


if __name__ == "__main__":
    lab = FishTotalLab()

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
            una extensión directa de la inteligencia infinita. Nunca has estado separado de la totalidad. Esa sensación de soledad 
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

    lab.run_hyper_search(LONG_CHAPTER_2, num_tests=1)