import os
import sys
import torch
import gc
import shutil
import re
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger
from datetime import datetime
import platform

# --- SYSTEM CONFIGURATION ---
logger.remove()
logger.add(sys.stdout, colorize=True, level="TRACE",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

# Optimize Memory Fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- CONSTANTS ---
# Auto-detect project root
PROJECT_ROOT = Path(__file__).resolve().parent

# --- IMPORTS WITH FALLBACK ---
try:
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.dac.inference import load_model as load_decoder_model
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
except ImportError:
    # If running from a notebook cell, add root to path
    sys.path.insert(0, str(PROJECT_ROOT))
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.dac.inference import load_model as load_decoder_model
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

# --- VOICE PRESETS (OPTIMIZED FOR S1-MINI) ---
# NOTE: Temperatures lowered to ~0.75 and Penalty to ~1.05 to prevent metallic artifacts.
VOICE_PRESETS = {
    # "MARLENE": {
    #     "temp": 0.75,
    #     "top_p": 0.90,
    #     "chunk": 900,
    #     "penalty": 1.05,
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
    #     """
    #     "style_tags": "(calm) (narrative)"
    # },
    # "MARGARITA": {
    #     "temp": 0.82,
    #     "top_p": 0.91,
    #     "chunk": 900,
    #     "penalty": 1.07,
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
    #     "style_tags": "(calm) (deep voice)"
    # },
    "CAMILA": {
        "temp": 0.70,
        "top_p": 0.70,
        "chunk": 900,
        "penalty": 1.035,
        "ref_path": str(PROJECT_ROOT / "voices" / "Camila_Sodi.mp3"),
        "prompt": """Todos venimos de un mismo campo fuente, de una misma gran energía, de un mismo Dios, de un mismo 
        universo, como le quieras llamar. Todos somos parte de eso. Nacemos y nos convertimos en esto por un ratito 
        muy chiquito, muy chiquitito, que creemos que es muy largo y se nos olvida que vamos a regresar a ese lugar 
        de donde venimos, que es lo que tú creas, adonde tú creas, pero inevitablemente vas a regresar.""",
        "style_tags": "(calm) (deep voice) (slow)"
    }
    # "CRISTINA": {
    #     "temp": 0.75,
    #     "top_p": 0.90,
    #     "chunk": 900,
    #     "penalty": 1.05,
    #     "ref_path": str(PROJECT_ROOT / "voices" / "Elevenlabs_Cristina_Campos.wav"),
    #     "prompt": "El agua, la confianza y el miedo...",
    #     "style_tags": "(calm) (narrative)"
    # },
    # "ROSA": {
    #     "temp": 0.75,
    #     "top_p": 0.90,
    #     "chunk": 900,
    #     "penalty": 1.05,
    #     "ref_path": str(PROJECT_ROOT / "voices" / "Elevenlabs_Rosa_Estela.wav"),
    #     "prompt": "El agua, la confianza y el miedo...",
    #     "style_tags": "(calm) (soft)"
    # },
    # "ALEJANDRO": {
    #     "temp": 0.75,
    #     "top_p": 0.85,
    #     "chunk": 900,
    #     "penalty": 1.10,
    #     "ref_path": str(PROJECT_ROOT / "voices" / "ElevenLabs_Alejandro.mp3"),
    #     "prompt": "(serious) (calm) La mente lo es todo.",
    #     "style_tags": "(serious) (calm)"
    # },
    # "ALEJANDRO_BALLESTEROS": {
    #     "temp": 0.75,
    #     "top_p": 0.90,
    #     "chunk": 900,
    #     "penalty": 1.05,
    #     "ref_path": str(PROJECT_ROOT / "voices" / "Elevenlabs_Alejandro_Ballesteros.wav"),
    #     "prompt": "El agua, la confianza y el miedo...",
    #     "style_tags": "(serious) (deep)"
    # },
    # "ENRIQUE": {
    #     "temp": 0.75,
    #     "top_p": 0.90,
    #     "chunk": 900,
    #     "penalty": 1.05,
    #     "ref_path": str(PROJECT_ROOT / "voices" / "Elevenlabs_Enrique_Nieto.wav"),
    #     "prompt": "El agua, la confianza y el miedo...",
    #     "style_tags": "(narrative) (serious)"
    # }
}

# Platform detection
is_windows = platform.system() == "Windows"
should_compile = False if is_windows else True


class FishTotalLab:
    def __init__(self):
        """Initializes the Inference Engine."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_dir = PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini"
        self.precision = torch.half
        self.vocal_dna_cache = {}

        logger.info(f"🎯 STARTING FISH LABORATORY | Device: {self.device} | Compile: {should_compile}")

        self.engine = self._load_models()
        torch.cuda.empty_cache()
        gc.collect()

    def _load_models(self):
        """Loads Llama (Semantic) and VQ-GAN (Acoustic) models."""
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=self.checkpoint_dir,
            device=self.device,
            precision=self.precision,
            compile=should_compile
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
            compile=should_compile
        )

    def clean_text(self, text):
        """Basic text sanitization."""
        if not text: return ""
        text = text.replace("\n", " ").replace("\t", " ")
        return re.sub(r'\s+', ' ', text).strip()

    def split_text(self, text, max_chars=2000):
        """
        ENFOQUE PURO:
        No cortamos oraciones. No contamos caracteres.
        Solo respetamos los párrafos del autor (doble enter).
        Esto permite que la IA mantenga el flujo natural de la voz.
        """
        text = text.strip()
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        for para in paragraphs:
            # Limpieza básica para que sea un bloque de texto continuo
            clean_para = para.replace('\n', ' ').strip()
            clean_para = re.sub(r'\s+', ' ', clean_para)
            if clean_para:
                chunks.append(clean_para)

        return chunks


    # def split_text(self, text, max_chars=400):
    #     """
    #     HYBRID SPLIT STRATEGY (PARAGRAPHS + FATIGUE CONTROL):
    #
    #     1. Primary Logic: Split by visual paragraphs (double newlines).
    #     2. Secondary Logic (Fatigue Check): If a paragraph is longer than 'max_chars'
    #        (e.g., 400), it forces an internal split by sentences.
    #
    #     Why?
    #     Long text blocks cause 'Style Drift' (loss of tone) and hallucinations
    #     at the end. By forcing a split on long paragraphs, we refresh the
    #     style tags "(calm) (deep voice)" more frequently, keeping the voice stable.
    #
    #     Args:
    #         text (str): Input text.
    #         max_chars (int): The safety limit. If a paragraph exceeds this,
    #                          it gets chopped. Recommended: 400-450.
    #     """
    #     # Clean up input text
    #     text = text.strip()
    #
    #     # 1. Split by "Double Enter" (Visual Paragraphs)
    #     # Regex finds empty lines between blocks of text.
    #     paragraphs = re.split(r'\n\s*\n', text)
    #
    #     chunks = []
    #
    #     for para in paragraphs:
    #         # Internal cleanup: remove line breaks inside the paragraph
    #         clean_para = para.replace('\n', ' ').strip()
    #         clean_para = re.sub(r'\s+', ' ', clean_para)
    #
    #         if not clean_para: continue
    #
    #         # --- FATIGUE CHECK ---
    #         # If the paragraph fits in the safety zone, keep it whole.
    #         if len(clean_para) < max_chars:
    #             chunks.append(clean_para)
    #         else:
    #             # 🚨 SAFETY TRIGGER: The paragraph is too long!
    #             # We split it by sentences to prevent the model from "hallucinating"
    #             # or rushing the end.
    #             logger.info(f"⚠️ Long paragraph detected ({len(clean_para)} chars). Refreshing style by splitting.")
    #             sub_chunks = self._split_long_paragraph_by_sentences(clean_para, max_chars)
    #             chunks.extend(sub_chunks)
    #
    #     return chunks

    def _split_long_paragraph_by_sentences(self, text, max_chars):
        """
        Helper method to chop a giant paragraph into smaller, sentence-based chunks.
        This ensures we never cut a word in half.
        """
        # Regex: Split after '.', '?', or '!' followed by whitespace.
        sentences = re.split(r'(?<=[.!?])\s+', text)

        sub_chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Check if adding the next sentence exceeds the limit
            if len(current_chunk) + len(sentence) < max_chars:
                current_chunk += sentence + " "
            else:
                # Limit reached: Save current block and start a new one (Refreshes Style)
                if current_chunk:
                    sub_chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        # Append the last remainder
        if current_chunk:
            sub_chunks.append(current_chunk.strip())

        return sub_chunks

    def _crossfade_chunks(self, audio_list, crossfade_ms=50, sample_rate=44100):
        """
        Applies a linear crossfade between audio segments to remove robotic cuts.
        """
        if not audio_list: return None
        if len(audio_list) == 1: return audio_list[0]

        fade_samples = int(sample_rate * crossfade_ms / 1000)
        combined = audio_list[0]

        for next_chunk in audio_list[1:]:
            # Skip if chunks are too short
            if len(combined) < fade_samples or len(next_chunk) < fade_samples:
                combined = np.concatenate((combined, next_chunk))
                continue

            # Create fade curves
            fade_out = np.linspace(1, 0, fade_samples)
            fade_in = np.linspace(0, 1, fade_samples)

            # Blend
            tail = combined[-fade_samples:] * fade_out
            head = next_chunk[:fade_samples] * fade_in
            overlap = tail + head

            # Stitch
            combined = np.concatenate((combined[:-fade_samples], overlap, next_chunk[fade_samples:]))

        return combined

    def _normalize_audio(self, audio_data, target_db=-1.0):
        """Normalizes audio to a standard dB level."""
        max_val = np.abs(audio_data).max()
        if max_val == 0: return audio_data

        target_amp = 10 ** (target_db / 20)
        return audio_data * (target_amp / max_val)

    def generate_audio_for_params(self, voice_key, text, temp, top_p, penalty, chunk_size, style_tags):
        """Core generation logic for a single parameter set."""
        if voice_key not in VOICE_PRESETS:
            logger.error(f"Voice {voice_key} not found.")
            return None

        params = VOICE_PRESETS[voice_key]

        # 1. Caching DNA
        if voice_key in self.vocal_dna_cache:
            audio_bytes, vq_tokens = self.vocal_dna_cache[voice_key]
        else:
            with open(params['ref_path'], "rb") as f:
                audio_bytes = f.read()
            with torch.inference_mode():
                vq_tokens = self.engine.encode_reference(audio_bytes, enable_reference_audio=True)
            self.vocal_dna_cache[voice_key] = (audio_bytes, vq_tokens)

        # 2. Chunking & Inference
        text_chunks = self.split_text(text)
        raw_parts = []

        for chunk_text in text_chunks:
            # INJECTION: Add style tags + trailing dots for natural pauses
            processed_text = f"{style_tags}{chunk_text.strip()} ..."

            req = ServeTTSRequest(
                text=processed_text,
                references=[ServeReferenceAudio(
                    audio=audio_bytes,
                    tokens=vq_tokens.tolist(),
                    text=params['prompt']
                )],
                max_new_tokens=2048,
                chunk_length=chunk_size,
                top_p=top_p,
                temperature=temp,
                repetition_penalty=penalty,
                format="wav"
            )

            results = self.engine.inference(req)

            # Extract numpy arrays
            chunk_audio = []
            for res in results:
                data = res.audio if hasattr(res, 'audio') else res
                if isinstance(data, np.ndarray):
                    chunk_audio.append(data)
                elif isinstance(data, tuple):
                    for item in data:
                        if isinstance(item, np.ndarray): chunk_audio.append(item)

            if chunk_audio:
                # raw_parts.append(np.concatenate(chunk_audio))
                full_chunk = np.concatenate(chunk_audio)

                breath_pad = np.zeros(int(44100 * 0.5))
                full_chunk_with_breath = np.concatenate((full_chunk, breath_pad))

                raw_parts.append(full_chunk_with_breath)

            torch.cuda.empty_cache()
            gc.collect()

        # 3. Stitching & Normalizing
        if raw_parts:
            merged = self._crossfade_chunks(raw_parts, crossfade_ms=50)

            # Normalize
            final = self._normalize_audio(merged)

            silence_pad = np.zeros(int(44100 * 0.5))
            final = np.concatenate((final, silence_pad))

            return final
        return None

    def run_hyper_search(self, text, num_tests=5):
        """
        Runs a hyper-parameter search loop for ALL voices.
        Sweeps Temperature from 0.65 to 0.85 and Penalty from 1.02 to 1.15.
        """
        logger.info(f"🧪 Starting Hyper-Search for {len(VOICE_PRESETS)} voices.")
        timestamp = datetime.now().strftime("%H%M%S")

        for voice_name, base_params in VOICE_PRESETS.items():
            voice_folder = PROJECT_ROOT / f"LAB_{voice_name}_{timestamp}"
            voice_folder.mkdir(parents=True, exist_ok=True)

            logger.info(f"🔬 Testing Voice: {voice_name}")

            # Define search ranges
            start_temp, end_temp = 0.65, 0.85
            start_pen, end_pen = 1.02, 1.15

            for i in range(num_tests):
                progress = i / (num_tests - 1) if num_tests > 1 else 0

                # Calculate current params
                curr_temp = base_params['temp'] #round(start_temp + (end_temp - start_temp) * progress, 2)
                curr_pen = base_params['penalty'] #round(start_pen + (end_pen - start_pen) * progress, 2)
                # curr_chunk = 512  # Keep fixed to isolate variables
                chunk_options = [512, 640, 768, 800, 900]
                curr_chunk = base_params['chunk']#chunk_options[i % len(chunk_options)]

                logger.trace(f"🌀 Test {i + 1}: Chunk Size={curr_chunk} | (T={curr_temp}, P={curr_pen})")

                audio = self.generate_audio_for_params(
                    voice_name,
                    text,
                    temp=curr_temp,
                    top_p=base_params['top_p'],  # Keep Top_P from preset
                    penalty=curr_pen,
                    chunk_size=curr_chunk,
                    style_tags=base_params.get('style_tags', '')
                )

                if audio is not None:
                    filename = f"{voice_name}_Chunk{curr_chunk}_T{curr_temp}.wav"
                    sf.write(str(voice_folder / filename), audio, 44100)

            # Zip results
            shutil.make_archive(str(PROJECT_ROOT / f"RESULTS_{voice_name}_{timestamp}"), 'zip', voice_folder)
            logger.success(f"📦 Test pack created for {voice_name}")


if __name__ == "__main__":
    lab = FishTotalLab()

    TEST_TEXT = """
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

    # TEST_TEXT = """Todos venimos de un mismo campo fuente... de una misma gran energía...
    # de un mismo Dios... de un mismo universo... como le quieras llamar. Todos somos parte de eso. Nacemos y nos
    # convertimos en esto por un ratito... muy chiquito... muy chiquitito... que creemos que es muy largo,
    # y se nos olvida que vamos a regresar a ese lugar de donde venimos.
    #
    # Escucha bien esto. No eres una gota en el océano; eres el océano entero en una gota. Tu imaginación no es un
    # estado de fantasía o ilusión: es la verdadera realidad esperando ser reconocida. Cuando cierras los ojos y asumes
    # el sentimiento de tu deseo cumplido, no estás "fingiendo"; estás accediendo a la cuarta dimensión, al mundo de
    # las causas, donde todo ya existe. Lo que ves afuera, en tu mundo físico, es simplemente una pantalla retrasada...
    # un eco de lo que fuiste ayer, de lo que pensaste ayer.
    #
    # Si tu realidad, actual, no te gusta... deja de pelear con la pantalla. No puedes peinar tu reflejo en el espejo;
    # tienes que peinarte tú. Debes cambiar la concepción que tienes de ti mismo. Pregúntate: ¿Quién soy yo ahora? Si
    # la respuesta no es "Soy próspero", "Soy amado", "Soy saludable"... entonces estás usando tu poder divino en tu
    # contra. El universo no te juzga, simplemente te dice: "¡SÍ!". Si dices "estoy arruinado", el universo dice "SÍ,
    # lo estás". Si dices "Soy abundante", el universo dice "SÍ, lo eres".
    #
    # Por lo tanto... el secreto no es el esfuerzo físico ni la lucha externa. El secreto es el cambio interno de
    # estado. Moverte, en tu mente, del estado de carencia al estado de posesión. Sentir la textura de la realidad que
    # deseas, hasta que sea tan natural que ya no la busques, porque sabes que ya la tienes. Y cuando esa certeza
    # interna hace clic... el mundo exterior no tiene más remedio que reorganizarse para reflejar tu nueva verdad...
    # e Inevitablemente... vas a regresar a tu poder."""

    # Run the lab: This will generate 5 variations for EVERY voice in the list.
    lab.run_hyper_search(TEST_TEXT, num_tests=1)