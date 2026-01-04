import os
import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger

# --- CONFIGURACIÓN DE LOGS MEGA TRACE ---
logger.remove()
logger.add(sys.stdout, colorize=True, level="TRACE",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

# --- AUDITORÍA DE HARDWARE (TRACE CRÍTICO) ---
try:
    import torchaudio

    cuda_available = torch.cuda.is_available()
    # Verificamos si torchaudio puede ver la GPU
    ta_backends = torchaudio.list_audio_backends()

    logger.trace(f"🎸 [TRACE] Torchaudio Version: {torchaudio.__version__}")
    logger.trace(f"🎮 [TRACE] CUDA en PyTorch: {cuda_available}")
    logger.trace(f"🔊 [TRACE] Backends de Audio: {ta_backends}")

    if cuda_available and torch.version.cuda:
        logger.success(f"🚀 TODO SINCRONIZADO: CUDA {torch.version.cuda} detectado.")
    else:
        logger.warning("⚠️ Torchaudio detectado pero CUDA parece no estar activo en este entorno.")
except ImportError:
    logger.critical("❌ [TRACE] Torchaudio no encontrado en el entorno de Poetry.")
    sys.exit(1)

# --- RESTO DEL SCRIPT OFICIAL 3 PASOS ---
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
        # PASO 1: Codificación con referencia habilitada
        logger.trace("🧬 [PASO 1] Extrayendo VQ Tokens con aceleración CUDA...")
        with open(ref_path, "rb") as f:
            audio_bytes = f.read()

        vq_tokens = self.engine.encode_reference(
            audio=audio_bytes,
            enable_reference_audio=True
        )

        # PASO 2 y 3: Generación
        logger.info("🎙️ Generando audio clonado...")
        req = ServeTTSRequest(
            text=text,
            references=[ServeReferenceAudio(tokens=vq_tokens.tolist(), text=prompt_text)],
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
            output_path = PROJECT_ROOT / "clonacion_final_cuda.wav"
            sf.write(str(output_path), final_audio, 44100)
            logger.success(f"✅ ¡ÉXITO! Audio generado en GPU: {output_path}")


if __name__ == "__main__":
    lab = FishCudaLab()
    lab.run_official_step_by_step(
        text="La mente es la causa de todo; produce la realidad del individuo, ¡con total claridad!",
        prompt_text="Agradezco que cada vez trabajo menos y gano más, estoy tan feliz y agradecida.",
        ref_path="/kaggle/working/fish-speech/voice_to_clone.wav"
    )