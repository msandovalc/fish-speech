import os
import sys
import torch
import shutil
import random
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger
from datetime import datetime

# --- LOGS TRACE ACTIVADOS ---
logger.remove()
logger.add(sys.stdout, colorize=True, level="TRACE",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ["EINX_FILTER_TRACEBACK"] = "false"

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio


class FishMegaLab:
    def __init__(self):
        self.device = "cuda"
        self.checkpoint_dir = PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini"
        self.precision = torch.half

        logger.info("🚀 CARGANDO MODELOS PARA BATERÍA DE 50 PRUEBAS...")
        self.engine = self._load_models()

    def _load_models(self):
        llama_queue = launch_thread_safe_queue(checkpoint_path=self.checkpoint_dir, device=self.device,
                                               precision=self.precision, compile=True)
        decoder_model = load_decoder_model(config_name="modded_dac_vq",
                                           checkpoint_path=self.checkpoint_dir / "codec.pth", device=self.device)
        return TTSInferenceEngine(llama_queue=llama_queue, decoder_model=decoder_model, precision=self.precision,
                                  compile=True)

    def run_50_variants(self, text: str, ref_path: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = PROJECT_ROOT / f"mega_test_{timestamp}"
        folder.mkdir(parents=True, exist_ok=True)

        with open(ref_path, "rb") as f:
            ref_bytes = f.read()

        logger.info(f"🧪 Iniciando generación de 50 variantes en: {folder}")

        for i in range(1, 51):
            # Generación aleatoria controlada de parámetros para cubrir todo el espectro
            t = round(random.uniform(0.3, 1.2), 2)
            p = round(random.uniform(0.5, 0.95), 2)
            penalty = round(random.uniform(1.0, 1.6), 2)
            chunk = random.choice([150, 250, 350, 500])

            name = f"V{i:02d}_T{t}_P{p}_Pen{penalty}_C{chunk}"
            logger.trace(f"🌀 [Variante {i}/50] -> {name}")

            req = ServeTTSRequest(
                text=text,
                references=[ServeReferenceAudio(audio=ref_bytes, text="")],
                max_new_tokens=1024,
                chunk_length=chunk,
                top_p=p,
                temperature=t,
                repetition_penalty=penalty,
                format="wav"
            )

            results = self.engine.inference(req)
            chunks = []
            sr = 44100

            for res in results:
                # Extracción profunda confirmada
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
                final_audio = np.concatenate(chunks)
                sf.write(str(folder / f"{name}.wav"), final_audio, sr)
                logger.debug(f"✅ Guardada {name}")

        # Empaquetado
        zip_name = f"resultados_mega_50_{timestamp}"
        shutil.make_archive(str(PROJECT_ROOT / zip_name), 'zip', folder)
        logger.success(f"🏁 ¡PROCESO TERMINADO! 50 variantes listas en {zip_name}.zip")


if __name__ == "__main__":
    lab = FishMegaLab()
    # Texto con mejor puntuación para forzar prosodia
    TEXTO = "La mente es la causa de todo; produce la realidad del individuo, ¡con total claridad!"
    REFERENCIA = "/kaggle/working/fish-speech/voice_to_clone.wav"
    lab.run_50_variants(TEXTO, REFERENCIA)