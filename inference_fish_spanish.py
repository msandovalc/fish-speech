import os
import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger

# --- CONFIGURACIÓN DE LOGS NIVEL TRACE (NO SE QUITA NADA) ---
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

        logger.info(f"🚀 HARDWARE: Tesla T4 | GPU: {torch.cuda.get_device_name(0)}")
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
        logger.info(f"📝 Texto: {text}")
        logger.info(f"🎤 Referencia: {ref_audio_path}")

        with open(ref_audio_path, "rb") as f:
            ref_audio_bytes = f.read()

        request = ServeTTSRequest(
            text=text,
            references=[ServeReferenceAudio(audio=ref_audio_bytes, text="")],
            max_new_tokens=1024,
            chunk_length=200,
            format="wav"
        )

        logger.info("🎙️ Iniciando Inferencia...")
        results = self.engine.inference(request)

        audio_chunks = []
        sr = 44100  # Valor por defecto inicial

        for i, res in enumerate(results):
            logger.trace(f"📦 [Chunk {i}] --- INICIO DE TRACE ---")
            logger.trace(f"   ∟ Tipo base recibido: {type(res)}")

            # 1. Extraer del objeto de inferencia
            chunk = res.audio if hasattr(res, 'audio') else res
            logger.trace(f"   ∟ Contenido de .audio/res: {type(chunk)}")

            # 2. Navegar la Tupla (SampleRate, Data)
            if isinstance(chunk, tuple):
                logger.trace(f"   ∟ [Tupla Detectada] Longitud: {len(chunk)}")
                for idx, item in enumerate(chunk):
                    logger.trace(f"      ∟ Índice [{idx}]: {type(item)}")
                    if isinstance(item, int):
                        sr = item
                        logger.debug(f"      🎯 Sample Rate extraído: {sr}")
                    elif isinstance(item, np.ndarray):
                        audio_chunks.append(item)
                        logger.trace(f"      ✅ Array de audio encontrado (Shape: {item.shape})")

            # 3. Si viene el Array directo
            elif isinstance(chunk, np.ndarray):
                audio_chunks.append(chunk)
                logger.trace(f"   ∟ Array directo encontrado (Shape: {chunk.shape})")

        if not audio_chunks:
            logger.critical("💀 ERROR: No se capturó ningún array de audio.")
            return

        # --- UNIÓN Y DIAGNÓSTICO DE SEÑAL ---
        logger.info(f"🧩 Uniendo {len(audio_chunks)} fragmentos...")

        # Concatenamos de forma nativa en Numpy
        final_audio = np.concatenate(audio_chunks)

        # LOGS DE TRACE PARA DIAGNÓSTICO DE "SILENCIO"
        logger.trace(f"📊 --- ESTADÍSTICAS DE AUDIO ---")
        logger.trace(f"   ∟ Tipo de dato (Dtype): {final_audio.dtype}")
        logger.trace(f"   ∟ Forma (Shape): {final_audio.shape}")
        logger.trace(f"   ∟ Valor Máximo: {np.max(final_audio)}")
        logger.trace(f"   ∟ Valor Mínimo: {np.min(final_audio)}")
        logger.trace(f"   ∟ Media (Amplitude): {np.mean(np.abs(final_audio))}")

        if np.max(np.abs(final_audio)) < 1e-5:
            logger.warning("⚠️ ALERTA: El audio parece estar casi en silencio absoluto.")

        output_path = PROJECT_ROOT / "clonacion_final_es.wav"

        # Soundfile maneja el dtype automáticamente al escribir
        sf.write(str(output_path), final_audio, sr)

        logger.success(f"🎊 ¡LOGRADO! Archivo guardado con éxito: {output_path}")


if __name__ == "__main__":
    tts = FishSpanishInference()
    TEXTO = "La mente es la causa de todo, produce la realidad del individuo con total claridad."
    REFERENCIA = "/kaggle/working/fish-speech/voice_to_clone.wav"
    tts.generate(TEXTO, REFERENCIA)