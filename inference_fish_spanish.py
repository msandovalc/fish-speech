import os
import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger

# --- CONFIGURACIÓN DE LOGS NIVEL TRACE ---
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

        logger.info(
            f"🚀 HARDWARE: Tesla T4 (Kaggle) | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        self.engine = self._load_models()

    def _load_models(self):
        logger.debug("🛰️ Cargando Llama Queue...")
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=self.checkpoint_dir,
            device=self.device,
            precision=self.precision,
            compile=True
        )
        logger.debug("🔊 Cargando Decoder VQ-GAN...")
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

        audio_parts = []

        for i, res in enumerate(results):
            logger.trace(f"📦 [Chunk {i}] Iniciando inspección profunda...")

            chunk = res
            depth = 0

            # Bucle de extracción recursivo con logging TRACE
            while True:
                depth += 1
                logger.trace(f"   ∟ [Nivel {depth}] Tipo actual: {type(chunk)}")

                # CASO A: Encontramos los bytes (Éxito)
                if isinstance(chunk, (bytes, bytearray)):
                    logger.debug(f"✅ [Chunk {i}] Bytes extraídos en Nivel {depth} ({len(chunk)} bytes)")
                    audio_parts.append(chunk)
                    break

                # CASO B: Es una tupla (Típico de InferenceResult.audio o retorno directo)
                elif isinstance(chunk, tuple):
                    logger.trace(f"     ∟ Detectada Tupla (len={len(chunk)}). Extrayendo índice [0]...")
                    if len(chunk) > 0:
                        chunk = chunk[0]
                    else:
                        logger.error(f"     ∟ Tupla vacía en Chunk {i}")
                        break

                # CASO C: Es un objeto con atributo .audio
                elif hasattr(chunk, 'audio'):
                    logger.trace(f"     ∟ Atributo '.audio' detectado. Bajando un nivel...")
                    chunk = chunk.audio

                # CASO D: Tipo desconocido
                else:
                    logger.error(f"❌ [Chunk {i}] Tipo no manejable en Nivel {depth}: {type(chunk)}")
                    break

        if not audio_parts:
            logger.critical("💀 ERROR: La lista de audio está vacía. No hay nada que unir.")
            return

        logger.info(f"🧩 Uniendo {len(audio_parts)} fragmentos de audio...")
        try:
            audio_data = b"".join(audio_parts)
            audio_np = np.frombuffer(audio_data, dtype=np.int16)

            output_path = PROJECT_ROOT / "clonacion_final_es.wav"
            sf.write(str(output_path), audio_np, 44100)
            logger.success(f"🎊 ¡LOGRADO! Archivo guardado: {output_path}")
        except Exception as e:
            logger.exception(f"💥 Error fatal al concatenar bytes: {e}")


if __name__ == "__main__":
    tts = FishSpanishInference()
    TEXTO = "La mente es la causa de todo, produce la realidad del individuo con total claridad."
    # Aseguramos ruta absoluta en Kaggle
    REFERENCIA = "/kaggle/working/fish-speech/voice_to_clone.wav"
    tts.generate(TEXTO, REFERENCIA)