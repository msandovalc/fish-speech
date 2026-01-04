import os
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger

# 1. DETECCIÓN DINÁMICA DE RUTA
# Esto detecta la carpeta donde está guardado este archivo .py
PROJECT_ROOT = Path(__file__).resolve().parent
os.environ["EINX_FILTER_TRACEBACK"] = "false"

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIO_PATH = PROJECT_ROOT / 'voice_to_clone.wav'

class FishSpanishInference:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 2. RUTA PORTABLE: Construimos la ruta desde el PROJECT_ROOT detectado
        self.checkpoint_dir = PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini"
        self.precision = torch.half

        logger.info(f"📂 Proyecto detectado en: {PROJECT_ROOT}")
        logger.info(f"🛰️ Buscando modelos en: {self.checkpoint_dir}")

        self.engine = self._load_models()

    def _load_models(self):
        # Verificación de existencia para evitar el FileNotFoundError
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"❌ No se encontró la carpeta de modelos en: {self.checkpoint_dir}")

        llama_queue = launch_thread_safe_queue(
            checkpoint_path=self.checkpoint_dir,
            device=self.device,
            precision=self.precision,
            compile=True  # Activado para Kaggle
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
        with open(ref_audio_path, "rb") as f:
            ref_audio_bytes = f.read()

        request = ServeTTSRequest(
            text=text,
            references=[ServeReferenceAudio(audio=ref_audio_bytes, text="")],
            max_new_tokens=1024,
            chunk_length=200,
            format="wav"
        )

        logger.info(f"🎙️ Sintetizando...")
        results = list(self.engine.inference(request))

        # FIX de tupla y bytes
        audio_parts = []
        for res in results:
            if isinstance(res, tuple):
                audio_parts.append(res[0])
            elif hasattr(res, 'audio'):
                audio_parts.append(res.audio)
            else:
                audio_parts.append(res)

        audio_data = b"".join(audio_parts)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)

        # Guardar el resultado en la raíz del proyecto
        final_output = PROJECT_ROOT / output_name
        sf.write(str(final_output), audio_np, 44100)
        logger.success(f"✅ ¡LOGRADO! Audio guardado en: {final_output}")


if __name__ == "__main__":
    tts = FishSpanishInference()

    # TEXTO DE PRUEBA
    MI_TEXTO = "La mente es la causa de todo, produce la realidad del individuo con total claridad."

    # IMPORTANTE: Asegúrate de que esta ruta apunte a tu audio en el input de Kaggle
    RUTA_REF = str(AUDIO_PATH)

    tts.generate(MI_TEXTO, RUTA_REF)