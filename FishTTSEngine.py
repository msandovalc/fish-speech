import os
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
PROJECT_ROOT = Path(__file__).resolve().parent

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
        "ref_path": "/kaggle/working/fish-speech/ElevenLabs_Marlene.mp3",
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
        "ref_path": f"{PROJECT_ROOT}/voices/Camila_Sodi.mp3",
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
        "ref_path": "/kaggle/working/fish-speech/ElevenLabs_Alejandro.mp3",
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

    def __init__(self):
        """Initializes the engine, detecting platform and loading models safely."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_dir = PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini"
        self.precision = torch.half

        # Windows doesn't support torch.compile (Triton missing)
        self.should_compile = False if platform.system() == "Windows" else True

        logger.info(f"🚀 Initializing Engine | OS: {platform.system()} | Compile: {self.should_compile}")

        try:
            self.engine = self._load_models()
            logger.success("✅ Models loaded successfully into VRAM.")
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            sys.exit(1)

    def _load_models(self):
        """Loads Llama and DAC models with memory safety."""
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

    def split_text(self, text, max_chars=1000):
        """Splits long text into chunks by sentences to avoid cutting words."""
        logger.debug(f"✂️ Splitting text into chunks (Max Chars: {max_chars})")
        # Split by periods to keep semantic meaning
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chars:
                current_chunk += sentence + ". "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        logger.info(f"📦 Text divided into {len(chunks)} logical chunks.")
        return chunks

    def process_narration(self, voice_key, full_text):
        """Main pipeline to process a full chapter of audio."""
        if voice_key not in VOICE_PRESETS:
            logger.error(f"❌ Voice '{voice_key}' not found in presets.")
            return

        params = VOICE_PRESETS[voice_key]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / f"narration_{voice_key}_{timestamp}"
        output_dir.mkdir(parents=True,
                         exist_ok=True
                         )

        logger.info(f"🎙️ Starting narration for {voice_key} | Total text length: {len(full_text)} chars.")

        try:
            # Step 1: Encode DNA Reference once
            with open(params['ref_path'], "rb") as f:
                audio_bytes = f.read()

            with torch.inference_mode():
                logger.trace(f"🧬 Encoding Vocal DNA for {voice_key}...")
                vq_tokens = self.engine.encode_reference(audio_bytes, enable_reference_audio=True)

            # Step 2: Split text
            text_chunks = self.split_text(full_text)
            final_audio_segments = []

            # Step 3: Process chunks in a loop
            for i, chunk in enumerate(text_chunks):
                logger.trace(f"⏳ Processing Chunk {i + 1}/{len(text_chunks)} | {len(chunk)} chars")

                req = ServeTTSRequest(
                    text=chunk,
                    references=[
                        ServeReferenceAudio(audio=audio_bytes,
                                            tokens=vq_tokens.tolist(),
                                            text=params['prompt']
                                            )
                    ],
                    max_new_tokens=2500,  # Increased for 1-2 min of audio per chunk
                    chunk_length=params['chunk'],
                    top_p=params['top_p'],
                    temperature=params['temp'],
                    repetition_penalty=params['penalty'],
                    format="wav"
                )

                # Inference
                results = self.engine.inference(req)

                # Audio Extraction
                for res in results:
                    audio_data = res.audio if hasattr(res, 'audio') else res
                    if isinstance(audio_data, np.ndarray):
                        final_audio_segments.append(audio_data)

                # Immediate VRAM Cleanup
                torch.cuda.empty_cache()
                gc.collect()

            # Step 4: Final Assembly
            if final_audio_segments:
                combined_audio = np.concatenate(final_audio_segments)
                final_filename = f"FULL_NARRATION_{voice_key}_{timestamp}.wav"
                final_path = output_dir / final_filename

                sf.write(str(final_path),
                         combined_audio,
                         samplerate=44100
                         )
                logger.success(f"🏆 NARRATION COMPLETE: {final_path}")

                # Create ZIP
                shutil.make_archive(str(PROJECT_ROOT / f"bundle_{voice_key}_{timestamp}"), 'zip', output_dir)
                logger.info(f"📦 Bundle ZIP created.")

        except Exception as e:
            logger.exception(f"🔥 Critical failure during narration: {e}")


# --- EXECUTION ---
if __name__ == "__main__":
    engine = FishTTSEngine()

    # Example: A long text that would normally crash or cut
    LONG_CHAPTER = """
    Pensamiento causal 

    La mente causa mediante el pensamiento, lo bueno y lo malo para el propio individuo y este es responsable por 
    ello. Pensar es causar, pensamos en todo momento y causamos siempre. De manera que, cuando sostenemos El 
    pensamiento del bien
    
    pensamientos que llegan a una conclusión, automáticamente manifestamos los efectos correspondientes en nuestro 
    cuerpo, o experiencias por es esto, deberíamos tener solo pensamientos de bien, para experimentar o manifestar 
    precisamente lo bueno.
    
    El gran secreto, muy sencillo y claro, por lo demás, para llegar al entendimiento y aplicación de la verdad es, 
    mantener nuestro pensamiento en el bien, en forma continua, que causará, invariablemente, y en forma automática, 
    todo lo bueno, las metas realmente importantes de la vida, buena salud, buen abastecimiento, buenas finanzas o 
    fortuna y felicidad.
    
    No olvidar el hecho siempre en operación, que indica que en la exacta proporción en que mantenemos un 
    pensamiento, así se manifiesta o realiza en nuestra experiencia. De manera que, habrá que aumentar la proporción 
    de buenos pensamientos para obtener la misma alta propor- ción de resultados buenos.
    
    A veces pensamos que nuestro pensamiento no causa, quizá porque queremos que no produzca algo en el momento dado 
    y hasta podemos creer que así es. La realidad es que causamos siempre, porque nunca dejamos de pensar; la única 
    posibilidad de no causar con nuestro pensamiento es cuando tenemos pensamientos superficiales y pasajeros, 
    que de ninguna manera llegan a una convicción o conclusión, de otra forma siempre es- taremos causando algo; de 
    aquí que deberíamos vigilar constantemente nuestro pensar y evitar el pensamiento erróneo.
    """

    # Run production for Marlene
    engine.process_narration(voice_key="CAMILA",
                             full_text=LONG_CHAPTER)