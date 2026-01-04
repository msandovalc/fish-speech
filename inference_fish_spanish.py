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

# --- CONFIGURACIÓN DE LOGS TRACE (MÁXIMA VISIBILIDAD) ---
logger.remove()
logger.add(sys.stdout, colorize=True, level="TRACE",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
PROJECT_ROOT = Path("/kaggle/working/fish-speech")

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio


class FishHyperRefinement:
    def __init__(self):
        self.device = "cuda"
        self.checkpoint_dir = PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini"
        self.precision = torch.half
        logger.info("🎯 MODO HYPER-REFINAMIENTO: BUSCANDO LA EXPRESIÓN PERFECTA")
        self.engine = self._load_models()

    def _load_models(self):
        llama_queue = launch_thread_safe_queue(checkpoint_path=self.checkpoint_dir, device=self.device,
                                               precision=self.precision, compile=True)
        decoder_model = load_decoder_model(config_name="modded_dac_vq",
                                           checkpoint_path=self.checkpoint_dir / "codec.pth", device=self.device)
        return TTSInferenceEngine(llama_queue=llama_queue, decoder_model=decoder_model, precision=self.precision,
                                  compile=False)

    def run_hyper_search(self, text, prompt_text, ref_path, num_tests=15):
        logger.trace(f"🧬 [PASO 1] Codificando ADN Vocal de FINA_08...")
        with open(ref_path, "rb") as f:
            audio_bytes = f.read()

        with torch.inference_mode():
            vq_tokens = self.engine.encode_reference(audio_bytes, enable_reference_audio=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = PROJECT_ROOT / f"hyper_paunel_{timestamp}"
        folder.mkdir(parents=True, exist_ok=True)

        # RANGOS DE PRECISIÓN (Basados en FINA_08: T=0.77, P=0.83, C=413)
        t_min, t_max = 0.77, 0.86  # Subimos para más expresión
        p_min, p_max = 0.83, 0.88  # Subimos para más agudeza/brillo

        for i in range(num_tests):
            progress = i / (num_tests - 1) if num_tests > 1 else 0
            curr_t = round(t_min + (0.09 * progress), 2)
            curr_p = round(p_min + (0.05 * progress), 2)
            # Chunks largos para máxima fluidez
            curr_chunk = int(500 + (250 * progress))
            curr_pen = 1.12  # Penalización baja para que la voz sea más natural

            logger.trace("-" * 50)
            logger.trace(f"🌀 [HYPER {i + 1}/{num_tests}] | T={curr_t} | P={curr_p} | Chunk={curr_chunk}")

            name = f"PAUNEL_FINAL_{i + 1:02d}_T{curr_t}_P{curr_p}_C{curr_chunk}"

            req = ServeTTSRequest(
                text=text,
                references=[ServeReferenceAudio(audio=audio_bytes, tokens=vq_tokens.tolist(), text=prompt_text)],
                max_new_tokens=1024,
                chunk_length=curr_chunk,
                top_p=curr_p,
                temperature=curr_t,
                repetition_penalty=curr_pen,
                format="wav"
            )

            try:
                results = self.engine.inference(req)
                audio_parts = [res.audio if hasattr(res, 'audio') else res for res in results]
                # Limpieza de fragmentos
                clean_parts = []
                for p in audio_parts:
                    if isinstance(p, tuple):
                        clean_parts.extend([x for x in p if isinstance(x, np.ndarray)])
                    elif isinstance(p, np.ndarray):
                        clean_parts.append(p)

                if clean_parts:
                    sf.write(str(folder / f"{name}.wav"), np.concatenate(clean_parts), 44100)
                    logger.debug(f"💾 Guardado: {name}.wav")

                if (i + 1) % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                logger.error(f"❌ Error: {e}")

        shutil.make_archive(str(PROJECT_ROOT / f"paunel_final_pack_{timestamp}"), 'zip', folder)
        logger.success("🏁 ¡BATERÍA HYPER TERMINADA!")


if __name__ == "__main__":
    lab = FishHyperRefinement()

    # TEXTO CON MÁXIMA EXPRESIÓN (Puntuación clave)
    TEXTO = "¡Atención! La mente es la causa de todo... ¿Lo entiendes? ¡Produce la realidad del individuo con total, total claridad!"

    PROMPT = """
    La mente lo es todo. 

    La causa mental.
    
    La causa de todo -absolutamente todo- es mental, es decir,  la mente es la que produce o causa todo en la vida del individuo.  
    Cuando reconozcamos, entendamos y aceptemos esta verdad,  habremos dado un paso muy importante en el progreso del desarrollo. 
    
    Si todo es mental, este es un universo mental, donde todo funciona por  medios mentales. Nosotros somos seres mentales, mentalidades buenas,  perfectas y eternas. 
    
    La mente sólo tiene una actividad, pensar. El pensamiento es todo lo  de la mente lo único que somos y tenemos es pensamiento, por ello, el  pensamiento es lo más importante de todo.
   
    """

    lab.run_hyper_search(TEXTO, PROMPT, "/kaggle/working/fish-speech/ElevenLabs_2026-01-04T19_56_14_Alejandro.mp3")