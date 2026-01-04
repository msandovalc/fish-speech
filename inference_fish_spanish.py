import os
import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger

# --- MANTENEMOS LOGS TRACE (PROHIBIDO QUITAR) ---
logger.remove()
logger.add(sys.stdout, colorize=True, level="TRACE",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

# --- VERIFICACIÓN DE HARDWARE ---
try:
    import torchaudio

    logger.trace(f"🎸 [TRACE] Torchaudio Version: {torchaudio.__version__}")
    logger.trace(f"🎮 [TRACE] CUDA: {torch.cuda.is_available()} (Version: {torch.version.cuda})")
except ImportError:
    logger.critical("❌ [TRACE] Torchaudio no encontrado.")
    sys.exit(1)

PROJECT_ROOT = Path("/kaggle/working/fish-speech")
os.environ["EINX_FILTER_TRACEBACK"] = "false"

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio


class FishCudaLab:
    def __init__(self):
        self.device = "cuda"
        self.checkpoint_dir = PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini"
        self.precision = torch.half
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

    def run_official_step_by_step(self, text, prompt_text, ref_path):
        # PASO 1: Codificación (Fix de argumento posicional)
        logger.trace(f"🧬 [PASO 1] Codificando referencia: {ref_path}")
        with open(ref_path, "rb") as f:
            audio_bytes = f.read()

        # FIX: Pasamos audio_bytes de forma posicional, no como keyword 'audio'
        vq_tokens = self.engine.encode_reference(
            audio_bytes,
            enable_reference_audio=True
        )
        logger.debug(f"✅ VQ Tokens listos. Shape: {vq_tokens.shape}")

        # PASO 2 y 3: Generación Semántica y Vocals
        logger.info("🎙️ Iniciando síntesis de voz...")
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

        results = self.engine.inference(req)
        audio_parts = []
        for res in results:
            chunk = res.audio if hasattr(res, 'audio') else res
            if isinstance(chunk, tuple):
                for item in chunk:
                    if isinstance(item, np.ndarray): audio_parts.append(item)
            elif isinstance(chunk, np.ndarray):
                audio_parts.append(chunk)

        if audio_parts:
            final_audio = np.concatenate(audio_parts)
            output_path = PROJECT_ROOT / "clonacion_final_oficial.wav"
            sf.write(str(output_path), final_audio, 44100)
            logger.success(f"✅ ¡LOGRADO! Audio generado: {output_path}")


if __name__ == "__main__":
    lab = FishCudaLab()
    # Tu texto de Neville Goddard / Desarrollo Personal
    TEXTO = "La mente es la causa de todo; produce la realidad del individuo, ¡con total claridad!"
    PROMPT = "Agradezco que cada vez trabajo menos y gano más, estoy tan feliz y agradecida."
    lab.run_official_step_by_step(TEXTO, PROMPT, "/kaggle/working/fish-speech/voice_to_clone.wav")