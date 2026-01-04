import os
import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger

# Configuración de logs para ver progreso en tiempo real en Kaggle
logger.remove()
logger.add(sys.stdout, colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", level="INFO")

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ["EINX_FILTER_TRACEBACK"] = "false"

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio


class FishSpanishInference:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_dir = PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini"
        self.precision = torch.half

        logger.info(
            f"🚀 HARDWARE: {self.device.upper()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
        self.engine = self._load_models()

    def _load_models(self):
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=self.checkpoint_dir,
            device=self.device,
            precision=self.precision,
            compile=True
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
            compile=True
        )

    def generate(self, text: str, ref_audio_path: str, output_name: str = "clonacion_final_es.wav"):
        # CARGAMOS EL AUDIO COMPLETO (Sin recortes para Kaggle)
        with open(ref_audio_path, "rb") as f:
            ref_audio_bytes = f.read()

        request = ServeTTSRequest(
            text=text,
            references=[ServeReferenceAudio(audio=ref_audio_bytes, text="")],
            max_new_tokens=1024,
            chunk_length=200,
            format="wav"
        )

        logger.info(f"🎙️ Iniciando síntesis con referencia completa...")
        results = list(self.engine.inference(request))

        # --- FIX CRÍTICO DE LA TUPLA ---
        audio_parts = []
        for res in results:
            # Si el motor devuelve (audio_bytes, metadata)
            if isinstance(res, tuple):
                audio_parts.append(res[0])
            # Si devuelve un objeto con atributo audio (versiones anteriores)
            elif hasattr(res, 'audio') and res.audio:
                audio_parts.append(res.audio)
            # Si ya son bytes
            elif isinstance(res, (bytes, bytearray)):
                audio_parts.append(res)

        if not audio_parts:
            logger.error("❌ No se recibieron datos de audio del motor.")
            return

        audio_data = b"".join(audio_parts)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)

        final_output = PROJECT_ROOT / output_name
        sf.write(str(final_output), audio_np, 44100)
        logger.success(f"✅ ¡LOGRADO! Archivo creado en: {final_output}")


if __name__ == "__main__":
    tts = FishSpanishInference()

    MI_TEXTO = "La mente es la causa de todo, produce la realidad del individuo con total claridad."

    # Asegúrate de que este archivo exista en la misma carpeta que el script
    RUTA_REF = str(PROJECT_ROOT / "voice_to_clone.wav")

    tts.generate(MI_TEXTO, RUTA_REF)