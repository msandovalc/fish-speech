import os
import sys
import torch
import gc
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger
from datetime import datetime

# --- CONFIGURACIÓN DE LOGS TRACE (PROHIBIDO QUITAR) ---
logger.remove()
logger.add(sys.stdout, colorize=True, level="TRACE",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

# --- OPTIMIZACIÓN DE MEMORIA CUDA ---
# Evita la fragmentación que causó el fallo de 174 MiB
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

PROJECT_ROOT = Path("/kaggle/working/fish-speech")

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio


class FishOfficialTraceLab:
    def __init__(self):
        self.device = "cuda"
        self.checkpoint_dir = PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini"
        self.precision = torch.half

        logger.info("🚀 CARGANDO MOTORES (AJUSTE DINÁMICO DE VRAM)")
        self.engine = self._load_models()

    def _load_models(self):
        logger.trace("📡 [TRACE] Cargando modelo semántico Llama con compilación...")
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=self.checkpoint_dir,
            device=self.device, precision=self.precision, compile=True
        )

        logger.trace("📡 [TRACE] Cargando Decoder DAC (Compilación desactivada para ahorrar 1.5GB)")
        # Desactivar compile aquí es lo que evita el OutOfMemoryError
        decoder_model = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=self.checkpoint_dir / "codec.pth",
            device=self.device
        )

        return TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=self.precision,
            compile=False  # Evita el overhead de memoria global
        )

    def run_official_step_by_step(self, text, prompt_text, ref_path):
        # Limpieza inicial
        gc.collect()
        torch.cuda.empty_cache()
        logger.trace(
            f"🧹 [TRACE] Caché de CUDA limpia. VRAM libre estimada: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")

        # PASO 1: Codificación de Referencia
        logger.trace(f"🧬 [PASO 1] Extrayendo VQ Tokens de: {ref_path}")
        with open(ref_path, "rb") as f:
            audio_bytes = f.read()

        # El modo inferencia ahorra memoria al no guardar gradientes
        with torch.inference_mode():
            vq_tokens = self.engine.encode_reference(
                audio_bytes,
                enable_reference_audio=True
            )
            logger.debug(f"✅ VQ Tokens listos. Shape: {vq_tokens.shape}")

            # PASO 2 y 3: Generación
            logger.info("🎙️ Iniciando síntesis oficial de 3 pasos...")
            req = ServeTTSRequest(
                text=text,
                references=[ServeReferenceAudio(
                    tokens=vq_tokens.tolist(),
                    text=prompt_text
                )],
                max_new_tokens=1024,
                chunk_length=500,
                top_p=0.8,
                temperature=0.7,
                format="wav"
            )

            # Monitoreo de memoria durante inferencia
            logger.trace(f"📊 [TRACE] VRAM usada antes de inferencia: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

            results = self.engine.inference(req)
            audio_parts = []
            for i, res in enumerate(results):
                logger.trace(f"📦 [TRACE] Recibido fragmento de audio {i}")
                chunk = res.audio if hasattr(res, 'audio') else res
                if isinstance(chunk, tuple):
                    for item in chunk:
                        if isinstance(item, np.ndarray): audio_parts.append(item)
                elif isinstance(chunk, np.ndarray):
                    audio_parts.append(chunk)

            if audio_parts:
                final_audio = np.concatenate(audio_parts)
                output_path = PROJECT_ROOT / "clonacion_final_oficial_trace.wav"
                sf.write(str(output_path), final_audio, 44100)
                logger.success(f"✅ ¡LOGRADO! Audio guardado en: {output_path}")
            else:
                logger.error("💀 [TRACE] La inferencia terminó pero no se generaron fragmentos.")


if __name__ == "__main__":
    lab = FishOfficialTraceLab()

    TEXTO = "La mente es la causa de todo; produce la realidad del individuo, ¡con total claridad!"
    PROMPT = "Agradezco que cada vez trabajo menos y gano más, estoy tan feliz y agradecida."
    REF = "/kaggle/working/fish-speech/voice_to_clone.wav"

    lab.run_official_step_by_step(TEXTO, PROMPT, REF)