import os
import sys
import torch
import gc
import shutil
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger
from datetime import datetime

# --- CONFIGURACIÓN DE PRESETS GLOBALES ---
# Aquí guardamos lo que ya funciona
VOICE_PRESETS = {
    "MARLENE": {
        "temp": 0.78,
        "top_p": 0.83,
        "chunk": 517,
        "penalty": 1.12
    }
}

# --- CONFIGURACIÓN DE LOGS TRACE ---
logger.remove()
logger.add(sys.stdout, colorize=True, level="TRACE",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
PROJECT_ROOT = Path("/kaggle/working/fish-speech")

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio


class FishVoiceLab:
    def __init__(self):
        self.device = "cuda"
        self.checkpoint_dir = PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini"
        self.precision = torch.half
        logger.info("🚀 LABORATORIO DE VOCES INICIADO - MODO BARRIDO GRAVE")
        self.engine = self._load_models()

    def _load_models(self):
        llama_queue = launch_thread_safe_queue(checkpoint_path=self.checkpoint_dir, device=self.device,
                                               precision=self.precision, compile=True)
        decoder_model = load_decoder_model(config_name="modded_dac_vq",
                                           checkpoint_path=self.checkpoint_dir / "codec.pth", device=self.device)
        return TTSInferenceEngine(llama_queue=llama_queue, decoder_model=decoder_model, precision=self.precision,
                                  compile=False)

    def run_hyper_search_male(self, text, prompt_text, ref_path, num_tests=15):
        logger.trace(f"🧬 [PASO 1] Extrayendo ADN Vocal (Hombre): {ref_path}")
        with open(ref_path, "rb") as f:
            audio_bytes = f.read()

        with torch.inference_mode():
            # Argumento posicional
            vq_tokens = self.engine.encode_reference(audio_bytes, enable_reference_audio=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = PROJECT_ROOT / f"search_male_{timestamp}"
        folder.mkdir(parents=True, exist_ok=True)

        # --- RANGOS PARA VOZ GRAVE (MÁS BAJOS PARA EVITAR DISTORSIÓN) ---
        t_start, t_end = 0.50, 0.75  # Voces graves prefieren T baja para estabilidad
        p_start, p_end = 0.82, 0.95  # P alta para claridad en la resonancia de pecho

        for i in range(num_tests):
            progress = i / (num_tests - 1) if num_tests > 1 else 0
            curr_t = round(t_start + ((t_end - t_start) * progress), 2)
            curr_p = round(p_start + ((p_end - p_start) * progress), 2)
            curr_chunk = int(450 + (300 * progress))  # Chunks largos para fluidez narrativa
            curr_pen = round(1.15 + (0.2 * progress), 2)  # Penalización incremental

            logger.trace("-" * 50)
            logger.trace(f"🌀 [CICLO {i + 1}/{num_tests}] | T={curr_t} | P={curr_p} | Pen={curr_pen}")

            name = f"ALEJANDRO_{i + 1:02d}_T{curr_t}_P{curr_p}_C{curr_chunk}"

            req = ServeTTSRequest(
                text=text,
                references=[ServeReferenceAudio(
                    audio=audio_bytes,
                    tokens=vq_tokens.tolist(),
                    text=prompt_text
                )],
                max_new_tokens=1024,
                chunk_length=curr_chunk,
                top_p=curr_p,
                temperature=curr_t,
                repetition_penalty=curr_pen,
                format="wav"
            )

            try:
                results = self.engine.inference(req)
                audio_parts = []
                for res in results:
                    chunk = res.audio if hasattr(res, 'audio') else res
                    if isinstance(chunk, tuple):
                        audio_parts.extend([x for x in chunk if isinstance(x, np.ndarray)])
                    elif isinstance(chunk, np.ndarray):
                        audio_parts.append(chunk)

                if audio_parts:
                    sf.write(str(folder / f"{name}.wav"), np.concatenate(audio_parts), 44100)
                    logger.debug(f"💾 Guardado: {name}.wav")

                if (i + 1) % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                logger.error(f"❌ Error en variante {i + 1}: {e}")

        shutil.make_archive(str(PROJECT_ROOT / f"alejandro_pack_{timestamp}"), 'zip', folder)
        logger.success(f"🏁 BARRIDO COMPLETADO. ZIP en: {folder}.zip")


if __name__ == "__main__":
    lab = FishVoiceLab()

    # TEXTO DE PRUEBA (Mantenemos la puntuación para ver expresión)
    TEXTO = "¡Atención! La mente es la causa de todo... ¿Lo entiendes? ¡Produce la realidad del individuo con total, total claridad!"

    # PROMPT DE ALEJANDRO (Asegúrate de que el audio diga exactamente esto)
    PROMPT = """La mente lo es todo. La causa mental. La causa de todo -absolutamente todo- es mental, es decir, la mente es la que produce o causa todo en la vida del individuo. Cuando reconozcamos, entendamos y aceptemos esta verdad, habremos dado un paso muy importante en el progreso del desarrollo. Si todo es mental, este es un universo mental, donde todo funciona por medios mentales. Nosotros somos seres mentales, mentalidades buenas, perfectas y eternas. La mente sólo tiene una actividad, pensar. El pensamiento es todo lo de la mente lo único que somos y tenemos es pensamiento, por ello, el pensamiento es lo más importante de todo."""

    lab.run_hyper_search_male(TEXTO, PROMPT, "/kaggle/working/fish-speech/ElevenLabs_2026-01-04T19_56_14_Alejandro.mp3")