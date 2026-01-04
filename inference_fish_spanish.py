import os
import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger

# --- CONFIGURACIÓN DE LOGS TOTAL ---
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

        logger.info(f"🚀 HARDWARE: Tesla T4 | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
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
            logger.trace(f"📦 [Chunk {i}] --- INICIO DE INSPECCIÓN ---")

            # Función interna de rescate con TRACE aumentado
            def extract_recursive(item, level=1):
                logger.trace(f"   ∟ [L{level}] Tipo: {type(item)}")

                # Éxito: Encontramos los bytes
                if isinstance(item, (bytes, bytearray)):
                    logger.debug(f"      ✅ Bytes encontrados! ({len(item)} bytes)")
                    return item

                # Caso: Tupla (Aquí estaba el fallo, ahora revisamos todos los elementos)
                if isinstance(item, tuple):
                    logger.trace(f"      📂 Tupla de {len(item)} elementos. Buscando bytes dentro...")
                    for idx, sub_item in enumerate(item):
                        logger.trace(f"         ∟ Probando índice [{idx}] (Tipo: {type(sub_item)})")
                        found = extract_recursive(sub_item, level + 1)
                        if found: return found

                # Caso: Objeto con atributo .audio
                if hasattr(item, 'audio'):
                    logger.trace(f"      🔎 Atributo '.audio' detectado.")
                    return extract_recursive(item.audio, level + 1)

                # Caso: El item es un numpy array (a veces viene así en lugar de bytes)
                if isinstance(item, np.ndarray):
                    logger.debug(f"      ⚠️ Detectado Numpy Array. Convirtiendo a bytes...")
                    return item.tobytes()

                return None

            chunk_bytes = extract_recursive(res)

            if chunk_bytes:
                audio_parts.append(chunk_bytes)
            else:
                logger.error(f"❌ [Chunk {i}] No se pudo extraer nada útil.")

        if not audio_parts:
            logger.critical("💀 ERROR: Secuencia vacía.")
            return

        logger.info(f"🧩 Uniendo {len(audio_parts)} fragmentos...")
        try:
            audio_data = b"".join(audio_parts)
            # Intentamos detectar si el buffer es int16 o float32
            audio_np = np.frombuffer(audio_data, dtype=np.int16)

            output_path = PROJECT_ROOT / "clonacion_final_es.wav"
            sf.write(str(output_path), audio_np, 44100)
            logger.success(f"🎊 ¡ÉXITO! Guardado en: {output_path}")
        except Exception as e:
            logger.exception(f"💥 Error en fase final: {e}")


if __name__ == "__main__":
    tts = FishSpanishInference()
    TEXTO = "La mente es la causa de todo, produce la realidad del individuo con total claridad."
    REFERENCIA = "/kaggle/working/fish-speech/voice_to_clone.wav"
    tts.generate(TEXTO, REFERENCIA)