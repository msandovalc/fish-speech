import os
import sys
import torch
import shutil
import random
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger
from datetime import datetime

# --- CONFIGURACIÓN DE LOGS MEGA TRACE ---
logger.remove()
logger.add(sys.stdout, colorize=True, level="TRACE",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

PROJECT_ROOT = Path("/kaggle/working/fish-speech")
os.environ["EINX_FILTER_TRACEBACK"] = "false"

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio


class FishOfficialStepLab:
    def __init__(self):
        self.device = "cuda"
        self.checkpoint_dir = PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini"
        self.precision = torch.half

        logger.info("🚀 CARGANDO MOTORES (12GB VRAM)")
        self.engine = self._load_models()

    def _load_models(self):
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=self.checkpoint_dir,
            device=self.device, precision=self.precision, compile=True
        )
        decoder_model = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=self.checkpoint_dir / "codec.pth",
            device=self.device
        )
        return TTSInferenceEngine(
            llama_queue=llama_queue, decoder_model=decoder_model,
            precision=self.precision, compile=True
        )

    def run_official_workflow(self, text, prompt_text, ref_path, num_tests=30):
        # --- PASO 1: Get VQ tokens from reference audio ---
        logger.trace(f"🧬 [PASO 1] Codificando referencia: {ref_path}")
        with open(ref_path, "rb") as f:
            audio_bytes = f.read()

        # Esto genera el equivalente al 'fake.npy' en memoria
        vq_tokens = self.engine.encode_reference(audio_bytes)
        logger.debug(f"✅ VQ Tokens listos (Shape: {vq_tokens.shape})")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = PROJECT_ROOT / f"pruebas_oficiales_{timestamp}"
        folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"🧪 Iniciando 50 variantes en: {folder}")

        for i in range(1, num_tests + 1):
            t = round(random.uniform(0.3, 1.0), 2)
            p = round(random.uniform(0.6, 0.95), 2)
            penalty = round(random.uniform(1.1, 1.4), 2)
            chunk = random.choice([250, 400, 600])

            name = f"V{i:02d}_T{t}_P{p}_Pen{penalty}_C{chunk}"
            logger.trace(f"🌀 [PASO 2 y 3] Generando variante {i}/50: {name}")

            # Implementación del Paso 2 y 3 oficiales
            req = ServeTTSRequest(
                text=text,
                references=[ServeReferenceAudio(
                    tokens=vq_tokens.tolist(),  # 'fake.npy' inyectado
                    text=prompt_text  # 'Your reference text'
                )],
                max_new_tokens=1024,
                chunk_length=chunk,
                top_p=p,
                temperature=t,
                repetition_penalty=penalty,
                format="wav"
            )

            try:
                results = self.engine.inference(req)
                audio_parts = []
                for res in results:
                    # Extracción profunda de audio (Fix para evitar TypeError)
                    chunk_data = res.audio if hasattr(res, 'audio') else res
                    if isinstance(chunk_data, tuple):
                        for item in chunk_data:
                            if isinstance(item, np.ndarray): audio_parts.append(item)
                    elif isinstance(chunk_data, np.ndarray):
                        audio_parts.append(chunk_data)

                if audio_parts:
                    final_audio = np.concatenate(audio_parts)
                    sf.write(str(folder / f"{name}.wav"), final_audio, 44100)
                    logger.debug(f"💾 Guardado: {name}.wav")

            except Exception as e:
                logger.error(f"❌ Error en variante {i}: {e}")

        # Empaquetado ZIP final
        zip_path = PROJECT_ROOT / f"resultados_clonacion_{timestamp}"
        shutil.make_archive(str(zip_path), 'zip', folder)
        logger.success(f"🏁 PROCESO FINALIZADO. Descarga el ZIP: {zip_path}.zip")


if __name__ == "__main__":
    lab = FishOfficialStepLab()

    TEXTO_A_GENERAR = "La mente es la causa de todo; produce la realidad del individuo, ¡con total claridad!"

    # EL TEXTO EXACTO DE TU AUDIO DE REFERENCIA
    TEXTO_DE_REFERENCIA = "Agradezco que cada vez trabajo menos y gano más, estoy tan feliz y agradecida."

    # RUTA DE TU AUDIO
    AUDIO_REF = "/kaggle/working/fish-speech/voice_to_clone.wav"

    lab.run_official_workflow(TEXTO_A_GENERAR, TEXTO_DE_REFERENCIA, AUDIO_REF)