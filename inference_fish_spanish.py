import os
import sys
import torch
import shutil
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger
from datetime import datetime

# --- CONFIGURACIÓN DE LOGS NIVEL TRACE (MÁXIMO DETALLE) ---
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

        logger.info("🚀 LABORATORIO: Cargando motores en Tesla T4...")
        self.engine = self._load_models()
        logger.success("✨ Motores listos para experimentación masiva.")

    def _load_models(self):
        logger.trace("📡 [TRACE] Iniciando carga de pesos de Llama...")
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=self.checkpoint_dir,
            device=self.device,
            precision=self.precision,
            compile=True
        )
        logger.trace("📡 [TRACE] Iniciando carga de Decoder VQ-GAN...")
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

    def run_lab_test(self, text: str, ref_path: str, configs: list):
        # 1. CREACIÓN DE CARPETA DE RESULTADOS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_folder = PROJECT_ROOT / f"pruebas_clonacion_{timestamp}"
        results_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"📁 Folder de experimentos creado: {results_folder}")
        logger.trace(f"📂 [TRACE] Ruta absoluta de trabajo: {results_folder.resolve()}")

        with open(ref_path, "rb") as f:
            ref_bytes = f.read()

        # 2. BUCLE DE GENERACIÓN
        for i, cfg in enumerate(configs):
            name = cfg['name']
            logger.info(f"🧪 [PROBANDO {i + 1}/{len(configs)}]: {name}")
            logger.trace(
                f"   ∟ Parámetros TRACE: T={cfg['temp']}, P={cfg['top_p']}, Penalty={cfg['penalty']}, Chunk={cfg['chunk']}")

            req = ServeTTSRequest(
                text=text,
                references=[ServeReferenceAudio(audio=ref_bytes, text="")],
                max_new_tokens=1024,
                chunk_length=cfg['chunk'],
                top_p=cfg['top_p'],
                temperature=cfg['temp'],
                repetition_penalty=cfg['penalty'],
                format="wav"
            )

            # Inferencia
            results = self.engine.inference(req)
            chunks = []
            sr = 44100

            for j, res in enumerate(results):
                # Usamos nuestra lógica de extracción profunda confirmada por los TRACE anteriores
                item = res.audio if hasattr(res, 'audio') else res
                if isinstance(item, tuple):
                    for sub in item:
                        if isinstance(sub, int):
                            sr = sub
                        elif isinstance(sub, np.ndarray):
                            chunks.append(sub)
                elif isinstance(item, np.ndarray):
                    chunks.append(item)

            if chunks:
                audio_final = np.concatenate(chunks)
                file_path = results_folder / f"{name}.wav"
                sf.write(str(file_path), audio_final, sr)
                logger.debug(f"💾 [TRACE] Guardado {name}.wav | Shape: {audio_final.shape}")
            else:
                logger.error(f"❌ Falló el experimento {name}")

        # 3. EMPAQUETADO ZIP
        logger.info("📦 Comprimiendo resultados para descarga...")
        zip_path = PROJECT_ROOT / f"resultados_{timestamp}"
        shutil.make_archive(str(zip_path), 'zip', results_folder)

        logger.success(f"🏁 BATERÍA FINALIZADA. Descarga tu archivo aquí: {zip_path}.zip")
        logger.trace(f"📦 [TRACE] Tamaño del ZIP: {os.path.getsize(f'{zip_path}.zip') / 1e6:.2f} MB")


if __name__ == "__main__":
    tts = FishSpanishInference()

    TEXTO = "La mente es la causa de todo; produce la realidad del individuo con total claridad."
    REFERENCIA = "/kaggle/working/fish-speech/voice_to_clone.wav"

    # --- DEFINICIÓN DE EXPERIMENTOS ---
    ajustes = [
        {"name": "01_Estable_Fiel", "temp": 0.5, "top_p": 0.7, "penalty": 1.2, "chunk": 300},
        {"name": "02_Natural_Variable", "temp": 0.7, "top_p": 0.8, "penalty": 1.1, "chunk": 250},
        {"name": "03_Narrador_Largo", "temp": 0.7, "top_p": 0.85, "penalty": 1.5, "chunk": 450},
        {"name": "04_Cerrado_Preciso", "temp": 0.3, "top_p": 0.5, "penalty": 1.1, "chunk": 300}
    ]

    tts.run_lab_test(TEXTO, REFERENCIA, ajustes)