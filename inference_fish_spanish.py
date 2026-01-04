import os
import sys
import torch
import gc
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger

# --- OPTIMIZACIÓN DE MEMORIA CUDA ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- CONFIGURACIÓN DE LOGS TRACE (PROHIBIDO QUITAR) ---
logger.remove()
logger.add(sys.stdout, colorize=True, level="TRACE",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

PROJECT_ROOT = Path("/kaggle/working/fish-speech")

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio


class FishFinalLab:
    def __init__(self):
        self.device = "cuda"
        self.checkpoint_dir = PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini"
        self.precision = torch.half

        logger.info("🚀 CARGANDO MOTORES (STABLE VRAM MODE)")
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
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=self.precision,
            compile=False
        )

    def run_official_step_by_step(self, text, prompt_text, ref_path):
        # Limpieza VRAM
        gc.collect()
        torch.cuda.empty_cache()

        # PASO 1: Codificación de Referencia
        logger.trace(f"🧬 [PASO 1] Codificando referencia: {ref_path}")
        with open(ref_path, "rb") as f:
            audio_bytes = f.read()

        with torch.inference_mode():
            # Extraemos los tokens (idéntico a dac/inference.py)
            vq_tokens = self.engine.encode_reference(
                audio_bytes,
                enable_reference_audio=True
            )
            logger.debug(f"✅ VQ Tokens listos. Shape: {vq_tokens.shape}")

            # PASO 2 y 3: Generación Semántica y Vocals
            logger.info("🎙️ Iniciando síntesis oficial de 3 pasos...")

            # FIX PYDANTIC: Incluimos 'audio=audio_bytes' para satisfacer el validador,
            # pero mantenemos 'tokens' para que el motor use el ADN pre-calculado
            req = ServeTTSRequest(
                text=text,
                references=[ServeReferenceAudio(
                    audio=audio_bytes,  # Requerido por el esquema
                    tokens=vq_tokens.tolist(),  # Tu ADN vocal extraído
                    text=prompt_text  # Tu referencia de texto
                )],
                max_new_tokens=1024,
                chunk_length=500,
                top_p=0.8,
                temperature=0.7,
                format="wav"
            )

            logger.trace(f"📊 [TRACE] VRAM disponible para síntesis: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")

            results = self.engine.inference(req)
            audio_parts = []
            for i, res in enumerate(results):
                logger.trace(f"📦 [TRACE] Generado fragmento {i}")
                chunk = res.audio if hasattr(res, 'audio') else res
                if isinstance(chunk, tuple):
                    for item in chunk:
                        if isinstance(item, np.ndarray): audio_parts.append(item)
                elif isinstance(chunk, np.ndarray):
                    audio_parts.append(chunk)

            if audio_parts:
                final_audio = np.concatenate(audio_parts)
                output_path = PROJECT_ROOT / "clonacion_perfectA.wav"
                sf.write(str(output_path), final_audio, 44100)
                logger.success(f"✅ ¡LOGRADO! Clonación finalizada: {output_path}")


if __name__ == "__main__":
    lab = FishFinalLab()

    TEXTO = "La mente es la causa de todo; produce la realidad del individuo, ¡con total claridad!"
    PROMPT = "Agradezco que cada vez trabajo menos y gano más, estoy tan feliz y agradecida."
    REF = "/kaggle/working/fish-speech/voice_to_clone.wav"

    lab.run_official_step_by_step(TEXTO, PROMPT, REF)