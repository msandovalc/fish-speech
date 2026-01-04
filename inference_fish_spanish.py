import os
import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger

# Configuración de logs inmediata
logger.remove()
logger.add(sys.stdout, colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ["EINX_FILTER_TRACEBACK"] = "false"

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio


class FishSpanishInference:
    def __init__(self):
        self.device = "cuda"
        self.checkpoint_dir = PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini"
        self.precision = torch.half

        logger.info(f"🚀 INICIANDO EN: {self.device.upper()} | GPU: {torch.cuda.get_device_name(0)}")
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

    def generate(self, text: str, ref_audio_path: str):
        with open(ref_audio_path, "rb") as f:
            ref_audio_bytes = f.read()

        request = ServeTTSRequest(
            text=text,
            references=[ServeReferenceAudio(audio=ref_audio_bytes, text="")],
            max_new_tokens=1024,
            chunk_length=200,
            format="wav"
        )

        logger.info("🎙️ Sintetizando...")
        # El motor devuelve un generador, lo convertimos a lista
        results = list(self.engine.inference(request))

        # EXTRACCIÓN FORZADA DE BYTES
        audio_parts = []
        for res in results:
            if isinstance(res, tuple):
                # Caso detectado en tus logs: (bytes, metadata)
                audio_parts.append(res[0])
            elif hasattr(res, 'audio'):
                # Caso objeto: res.audio
                audio_parts.append(res.audio)
            else:
                # Caso bytes puros
                audio_parts.append(res)

        # Unión final de bytes
        audio_data = b"".join(audio_parts)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)

        output_path = PROJECT_ROOT / "clonacion_final_es.wav"
        sf.write(str(output_path), audio_np, 44100)
        logger.success(f"✅ ¡LOGRADO! Archivo: {output_path}")


if __name__ == "__main__":
    tts = FishSpanishInference()
    TEXTO = "La mente es la causa de todo, produce la realidad del individuo con total claridad."
    # Ruta absoluta al archivo que ya confirmamos que mide 16.09s
    REFERENCIA = "/kaggle/working/fish-speech/voice_to_clone.wav"
    tts.generate(TEXTO, REFERENCIA)