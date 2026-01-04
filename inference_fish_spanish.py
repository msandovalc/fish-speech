import os
import sys
import torch
import random
import shutil
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger
from datetime import datetime

# --- CONFIGURACIÓN DE LOGS TRACE (OBLIGATORIO) ---
logger.remove()
logger.add(sys.stdout, colorize=True, level="TRACE",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

# Optimización VRAM para no superar los 14.74 GB de la T4
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
PROJECT_ROOT = Path("/kaggle/working/fish-speech")

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio


class FishOfficialLab:
    def __init__(self):
        self.device = "cuda"
        self.checkpoint_dir = PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini"
        self.precision = torch.half

        logger.info("🚀 CARGANDO MOTORES PARA CLONACIÓN OFICIAL")
        self.engine = self._load_models()

    def _load_models(self):
        # Paso 2: Modelo Semántico (Llama)
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=self.checkpoint_dir,
            device=self.device, precision=self.precision, compile=True
        )
        # Paso 1 y 3: Modelo de Audio (DAC)
        decoder_model = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=self.checkpoint_dir / "codec.pth",
            device=self.device
        )
        return TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=self.precision,
            compile=False  # Ahorro de VRAM detectado en pruebas previas
        )

    def run_50_variants(self, text, prompt_text, ref_path):
        # --- PASO 1 OFICIAL: Get VQ tokens de la referencia ---
        logger.trace(f"🧬 [PASO 1] Codificando ADN de voz: {ref_path}")
        with open(ref_path, "rb") as f:
            audio_bytes = f.read()

        with torch.inference_mode():
            # Extraemos los tokens inyectando el audio
            vq_tokens = self.engine.encode_reference(
                audio_bytes,
                enable_reference_audio=True
            )
            logger.debug(f"✅ VQ Tokens listos. Shape: {vq_tokens.shape}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = PROJECT_ROOT / f"mega_batch_paunel_{timestamp}"
        folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"🧪 Iniciando 50 pruebas con tu Prompt de abundancia...")

        for i in range(1, 51):
            # Variaciones para encontrar tu tono real
            t = round(random.uniform(0.6, 0.9), 2)
            p = round(random.uniform(0.7, 0.85), 2)
            penalty = round(random.uniform(1.1, 1.3), 2)
            chunk = random.choice([300, 500, 700])

            name = f"V{i:02d}_T{t}_P{p}_Pen{penalty}_C{chunk}"
            logger.trace(f"🌀 [PASO 2 y 3] Variante {i}/50: {name}")

            # --- PASO 2 Y 3 OFICIALES ---
            req = ServeTTSRequest(
                text=text,
                references=[ServeReferenceAudio(
                    audio=audio_bytes,  # Obligatorio por esquema Pydantic
                    tokens=vq_tokens.tolist(),  # Tu ADN vocal
                    text=prompt_text  # Tu transcripción exacta
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
                    chunk_data = res.audio if hasattr(res, 'audio') else res
                    if isinstance(chunk_data, tuple):
                        for item in chunk_data:
                            if isinstance(item, np.ndarray): audio_parts.append(item)
                    elif isinstance(chunk_data, np.ndarray):
                        audio_parts.append(chunk_data)

                if audio_parts:
                    final_audio = np.concatenate(audio_parts)
                    sf.write(str(folder / f"{name}.wav"), final_audio, 44100)
                    logger.debug(f"💾 {name}.wav guardado.")

            except Exception as e:
                logger.error(f"❌ Error en variante {i}: {e}")

        # ZIP de resultados
        zip_path = PROJECT_ROOT / f"clonacion_final_paunel_{timestamp}"
        shutil.make_archive(str(zip_path), 'zip', folder)
        logger.success(f"🏁 BATERÍA FINALIZADA. ZIP: {zip_path}.zip")


if __name__ == "__main__":
    lab = FishOfficialLab()

    TEXTO_NUEVO = "La mente es la causa de todo; produce la realidad del individuo, ¡con total claridad!"

    # TU TRANSCRIPCIÓN EXACTA (LA LLAVE)
    PROMPT_TEXT = ("Agradezco que cada vez trabajo menos y gano más, estoy tan feliz y agradecida "
                   "ahora que el dinero viene a mi en cantidades cada vez mayores de diversar "
                   "fuentes de forma continua y correcta, soy abundante.")

    REF_AUDIO = "/kaggle/working/fish-speech/voice_to_clone.wav"

    lab.run_50_variants(TEXTO_NUEVO, PROMPT_TEXT, REF_AUDIO)