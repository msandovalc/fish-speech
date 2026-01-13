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
        "temp": 0.82,
        "top_p": 0.91,
        "chunk": 807,
        "penalty": 1.07,
        "ref_path": str(PROJECT_ROOT / "voices" / "ElevenLabs_Marlene.mp3"),
        "prompt": """La mente lo es todo. La causa mental. La causa de todo -absolutamente todo- es mental, es decir, 
        la mente es la que produce o causa todo en la vida del individuo.

        Cuando reconozcamos, entendamos y aceptemos esta verdad, habremos dado un paso muy importante en el progreso del desarrollo. 

        Si todo es mental, este es un universo mental, donde todo funciona por medios mentales. Nosotros somos seres 
        mentales, mentalidades buenas, perfectas y eternas.

        La mente sólo tiene una actividad, pensar. El pensamiento es todo lo de la mente lo único que somos y tenemos es 
        pensamiento, por ello, el pensamiento es lo más importante de todo. 
        """
    },
    "CAMILA": {
        "temp": 0.82,
        "top_p": 0.91,
        "chunk": 807,
        "penalty": 1.07,
        "ref_path": str(PROJECT_ROOT / "voices" / "Camila_Sodi.mp3"),
        "prompt": """Todos venimos de un mismo campo fuente, de una misma gran energía, de un mismo Dios, de un mismo 
        universo, como le quieras llamar. Todos somos parte de eso. Nacemos y nos convertimos en esto por un ratito 
        muy chiquito, muy chiquitito, que creemos que es muy largo y se nos olvida que vamos a regresar a ese lugar 
        de donde venimos, que es lo que tú creas, adonde tú creas, pero inevitablemente vas a regresar."""
    },
    "ALEJANDRO": {
        "temp": 0.84,
        "top_p": 0.91,
        "chunk": 785,
        "penalty": 1.07,
        "ref_path": str(PROJECT_ROOT / "voices" / "ElevenLabs_Alejandro.mp3"),
        "prompt": """La mente lo es todo. La causa mental. La causa de todo -absolutamente todo- es mental, es decir, 
        la mente es la que produce o causa todo en la vida del individuo.

        Cuando reconozcamos, entendamos y aceptemos esta verdad, habremos dado un paso muy importante en el progreso del desarrollo. 

        Si todo es mental, este es un universo mental, donde todo funciona por medios mentales. Nosotros somos seres 
        mentales, mentalidades buenas, perfectas y eternas.

        La mente sólo tiene una actividad, pensar. El pensamiento es todo lo de la mente lo único que somos y tenemos es 
        pensamiento, por ello, el pensamiento es lo más importante de todo. 
        """
    }
}


class FishTTSEngine:
    def __init__(self, checkpoint_path=None):
        """
        Initializes the TTS engine.
        :param checkpoint_path: Optional custom path to models. Defaults to PROJECT_ROOT/checkpoints.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Cache to store encoded Vocal DNA
        # format: { "MARLENE": (audio_bytes, vq_tokens), ... }
        self.vocal_dna_cache = {}

        self.checkpoint_dir = checkpoint_path or (PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini")
        self.precision = torch.half

        # Windows does not support torch.compile due to missing Triton support
        self.should_compile = False if platform.system() == "Windows" else True

        logger.info(f"🚀 Initializing Engine | OS: {platform.system()} | Device: {self.device}")

        try:
            self.engine = self._load_models()
            logger.success("✅ Models loaded successfully into VRAM.")
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise e

    def _load_models(self):
        """Loads Llama (LLM) and DAC (Decoder) models with memory safety."""
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
        """Removes extra whitespaces, tabs, and newlines to prevent AI artifacts."""
        if not text: return ""
        text = text.replace("\n", " ").replace("\t", " ")
        return re.sub(r'\s+', ' ', text).strip()

    def split_text(self, text, max_chars=1000):
        """Splits long text into semantic chunks to avoid VRAM overflow."""
        sentences = text.replace('\n', ' ').split('. ')
        chunks, current_chunk = [], ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chars:
                current_chunk += sentence + ". "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def process_narration(self, voice_key, raw_text):
        """
        API-like method: Processes text and returns the raw audio data.
        NO files are saved within the engine directory.

        :param voice_key: The key for the voice (e.g., 'CAMILA').
        :param raw_text: The string to be narrated.
        :return: Tuple (numpy_array_audio, sample_rate)
        """
        if voice_key not in VOICE_PRESETS:
            logger.error(f"❌ Voice key '{voice_key}' not found in presets.")
            return None, None

        params = VOICE_PRESETS[voice_key]
        clean_input = self.clean_text(raw_text)
        final_audio_segments = []

        try:

            # --- STEP 1: VOCAL DNA (With Caching Logic) ---
            if voice_key in self.vocal_dna_cache:
                logger.info(f"⚡ Using cached Vocal DNA for: {voice_key}")
                audio_bytes, vq_tokens = self.vocal_dna_cache[voice_key]
            else:
                logger.info(f"🧬 First run: Encoding Reference for {voice_key}...")
                with open(params['ref_path'], "rb") as f:
                    audio_bytes = f.read()

                with torch.inference_mode():
                    vq_tokens = self.engine.encode_reference(audio_bytes,
                                                             enable_reference_audio=True
                                                             )

                # Store in cache for future calls
                self.vocal_dna_cache[voice_key] = (audio_bytes, vq_tokens)
                logger.success(f"💾 Vocal DNA for {voice_key} stored in cache.")

            # --- Step 2: Inference Loop  ---
            text_chunks = self.split_text(clean_input)
            for i, chunk_text in enumerate(text_chunks):
                logger.debug(f"⏳ Processing chunk {i + 1}/{len(text_chunks)}")

                req = ServeTTSRequest(
                    text=chunk_text,
                    references=[ServeReferenceAudio(
                        audio=audio_bytes,
                        tokens=vq_tokens.tolist(),
                        text=params['prompt']
                    )],
                    max_new_tokens=2500,
                    chunk_length=params['chunk'],
                    top_p=params['top_p'],
                    temperature=params['temp'],
                    repetition_penalty=params['penalty'],
                    format="wav"
                )

                # Collect audio segments in memory
                results = self.engine.inference(req)
                for res in results:
                    data = res.audio if hasattr(res, 'audio') else res
                    if isinstance(data, np.ndarray):
                        final_audio_segments.append(data)
                    elif isinstance(data, tuple):
                        for item in data:
                            if isinstance(item, np.ndarray):
                                final_audio_segments.append(item)

                # Memory management for Quadro T1000
                torch.cuda.empty_cache()
                gc.collect()

            if final_audio_segments:
                combined_audio = np.concatenate(final_audio_segments)
                return combined_audio, 44100

            return None, None

        except Exception as e:
            logger.error(f"🔥 Engine Error: {e}")
            return None, None


# --- EXECUTION ---
if __name__ == "__main__":
    engine = FishTTSEngine()

    # Example: A long text that would normally crash or cut
    # LONG_CHAPTER = """
    # Pensamiento causal...
    #
    # La mente causa mediante el pensamiento, lo bueno y lo malo para el propio individuo y este es responsable por
    # ello. Pensar es causar, pensamos en todo momento y causamos siempre. De manera que, cuando sostenemos pensamiento del bien
    # pensamientos que llegan a una conclusión, automáticamente manifestamos los efectos correspondientes en nuestro cuerpo, o
    # experiencias por es esto, deberíamos tener solo pensamientos de bien, para experimentar o manifestar precisamente lo bueno.
    #
    # El gran secreto, muy sencillo y claro, por lo demás, para llegar al entendimiento y aplicación de la verdad es, mantener nuestro
    # pensamiento en el bien, en forma continua, que causará, invariablemente, y en forma automática, todo lo bueno, las metas realmente
    # importantes de la vida, buena salud, buen abastecimiento, buenas finanzas o fortuna y felicidad.
    #
    # No olvidar el hecho siempre en operación, que indica que en la exacta proporción en que mantenemos un pensamiento, así se
    # manifiesta o realiza en nuestra experiencia. De manera que, habrá que aumentar la proporción de buenos pensamientos para
    # obtener la misma alta propor- ción de resultados buenos.
    #
    # A veces pensamos que nuestro pensamiento no causa, quizá porque queremos que no produzca algo en el momento dado y hasta podemos
    # creer que así es. La realidad es que causamos siempre, porque nunca dejamos de pensar; la única posibilidad de no causar con
    # nuestro pensamiento es cuando tenemos pensamientos superficiales y pasajeros, que de ninguna manera llegan a una convicción o
    # conclusión, de otra forma siempre es- taremos causando algo; de aquí que deberíamos vigilar constantemente nuestro pensar y
    # evitar el pensamiento erróneo.
    # """

    LONG_CHAPTER = """
    (sighing) (shouting) El gran secreto (sighing) muy sencillo y claro (whispering) es mantener nuestro pensamiento en el bien. 
    (proud) (excited) Invariablemente y en forma automática todo lo bueno llegará a tu vida (excited) .
    """

    # Run production for Camila
    engine.process_narration(voice_key="CAMILA",
                             raw_text=LONG_CHAPTER
                             )