import os
import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger

# --- NUEVA CONFIGURACIÓN DE LOGS PARA KAGGLE ---
logger.remove() # Eliminar configuración por defecto
logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", level="INFO")

# Detección de ruta
PROJECT_ROOT = Path(__file__).resolve().parent
os.environ["EINX_FILTER_TRACEBACK"] = "false"

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

# Corregimos AUDIO_PATH para que sea absoluto basado en el script
AUDIO_PATH = PROJECT_ROOT / 'voice_to_clone.wav'

class FishSpanishInference:
    def __init__(self):
        # Reporte explícito de Hardware
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 DISPOSITIVO DETECTADO: {self.device.upper()}")
        if self.device == "cuda":
            logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"💾 VRAM Disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        self.checkpoint_dir = PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini"
        self.precision = torch.half

        logger.info(f"📂 Proyecto: {PROJECT_ROOT}")
        self.engine = self._load_models()

    def _load_models(self):
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"❌ Sin modelos en: {self.checkpoint_dir}")

        logger.info("⏳ Cargando Llama (esto puede tardar unos minutos en T4)...")
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=self.checkpoint_dir,
            device=self.device,
            precision=self.precision,
            compile=True
        )

        logger.info("🔊 Cargando Decoder VQ-GAN...")
        decoder_model = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=self.checkpoint_dir / "codec.pth",
            device=self.device,
        )

        return TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=self.precision,
            compile=True
        )

    def generate(self, text: str, ref_audio_path: str, output_name: str = "clonacion_final_es.wav"):
        # Ya no recortamos el audio, lo cargamos completo
        with open(ref_audio_path, "rb") as f:
            ref_audio_bytes = f.read()

        request = ServeTTSRequest(
            text=text,
            references=[ServeReferenceAudio(audio=ref_audio_bytes, text="")],
            max_new_tokens=1024,
            chunk_length=200,
            format="wav"
        )

        logger.info(f"🎙️ Sintetizando con referencia completa ({ref_audio_path})...")
        results = list(self.engine.inference(request))

        # --- AQUÍ ESTÁ EL ARREGLO ---
        audio_parts = []
        for res in results:
            # Si el motor devuelve una tupla (audio, metadata)
            if isinstance(res, tuple):
                audio_parts.append(res[0])
                # Si devuelve un objeto con atributo audio
            elif hasattr(res, 'audio') and res.audio:
                audio_parts.append(res.audio)
            # Si ya son bytes puros
            elif isinstance(res, (bytes, bytearray)):
                audio_parts.append(res)

        if not audio_parts:
            logger.error("❌ No se generó ningún fragmento de audio.")
            return

        audio_data = b"".join(audio_parts)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)

        # Guardar resultado
        final_output = PROJECT_ROOT / output_name
        sf.write(str(final_output), audio_np, 44100)
        logger.success(f"✅ ¡ÉXITO! Audio guardado en: {final_output}")


if __name__ == "__main__":
    tts = FishSpanishInference()
    MI_TEXTO = "La mente es la causa de todo, produce la realidad del individuo con total claridad."
    # Usar el AUDIO_PATH dinámico
    tts.generate(MI_TEXTO, str(AUDIO_PATH))