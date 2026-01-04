import os
import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger

# --- MANTENEMOS LOGS TRACE ---
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    level="TRACE",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
)

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

        logger.info(f"🚀 HARDWARE: Tesla T4 | Optimizando para Fidelidad de Voz")
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
        logger.info(f"🎤 Usando referencia de {ref_audio_path}")

        with open(ref_audio_path, "rb") as f:
            ref_audio_bytes = f.read()

        # --- AJUSTES DE CALIDAD Y SIMILITUD ---
        # Temperature: 0.7 (Equilibrio entre estabilidad y emoción)
        # Top_p: 0.8 (Filtra opciones poco probables que causan mala pronunciación)
        # Repetition Penalty: 1.2 (Evita que la voz se vuelva robótica o tartamuda)
        request = ServeTTSRequest(
            text=text,
            references=[ServeReferenceAudio(audio=ref_audio_bytes, text="")],
            max_new_tokens=1024,
            chunk_length=300,  # Aumentado de 200 a 300 para mejor prosodia (ritmo)
            top_p=0.8,
            temperature=0.7,
            repetition_penalty=1.2,
            format="wav"
        )

        logger.info("🎙️ Iniciando Inferencia de alta fidelidad...")
        results = self.engine.inference(request)

        audio_chunks = []
        sr = 44100

        for i, res in enumerate(results):
            logger.trace(f"📦 [Chunk {i}] Inspeccionando...")

            chunk = res.audio if hasattr(res, 'audio') else res

            if isinstance(chunk, tuple):
                for item in chunk:
                    if isinstance(item, int):
                        sr = item
                    elif isinstance(item, np.ndarray):
                        audio_chunks.append(item)
            elif isinstance(chunk, np.ndarray):
                audio_chunks.append(chunk)

        if not audio_chunks:
            logger.critical("💀 ERROR: No se generó audio.")
            return

        final_audio = np.concatenate(audio_chunks)

        # Logs de diagnóstico de señal
        logger.trace(f"📊 --- TRACE DE SEÑAL ---")
        logger.trace(f"   ∟ Max Amplitud: {np.max(np.abs(final_audio))}")
        logger.trace(f"   ∟ Dtype: {final_audio.dtype}")

        output_path = PROJECT_ROOT / "clonacion_final_es.wav"
        sf.write(str(output_path), final_audio, sr)
        logger.success(f"🎊 ¡LOGRADO! Archivo optimizado: {output_path}")


if __name__ == "__main__":
    tts = FishSpanishInference()

    # TEXTO CON PUNTUACIÓN (Ayuda a la entonación)
    MI_TEXTO = "La mente es la causa de todo; produce la realidad del individuo con total claridad. ¡Cree en tu poder!"

    REFERENCIA = "/kaggle/working/fish-speech/voice_to_clone.wav"
    tts.generate(MI_TEXTO, REFERENCIA)