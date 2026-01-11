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
import platform

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

# --- CONFIGURACIÓN DE LOGS TRACE (PROTEGIDOS) ---
logger.remove()
logger.add(sys.stdout, colorize=True, level="TRACE",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- Constants for Directory Paths ---
# PROJECT_ROOT = Path("/kaggle/working/fish-speech")
PROJECT_ROOT = Path(__file__).resolve().parent

# --- LISTA GLOBAL DE PRESETS (ACTUALIZADA CON GANADORES) ---
VOICE_PRESETS = {
    # "MARLENE": {
    #     "temp": 0.82,
    #     "top_p": 0.91,
    #     "chunk": 807,
    #     "penalty": 1.07,
    #     "ref_path": f"{PROJECT_ROOT}/voices/ElevenLabs_Marlene_optimized.mp3",
    #     "prompt": "La mente lo es todo. La causa mental. La causa de todo -absolutamente todo- es mental, es decir, "
    #               "la mente es la que produce o causa todo en la vida del individuo."
    # },
    "MARGARITA": {
        "temp": 0.82,
        "top_p": 0.91,
        "chunk": 807,
        "penalty": 1.07,
        "ref_path": f"{PROJECT_ROOT}/voices/margarita_am_2025_v2.wav",
        "prompt": """Mira te comparto, hicimos tres cuartos más y no suelta todavía el sistema y otros detallitos,
        pero mira lo que te quiero comentar es que sé que suena raro, sé que se requiere"""
    },
    "Camila": {
        "temp": 0.82,
        "top_p": 0.91,
        "chunk": 807,
        "penalty": 1.07,
        "ref_path": f"{PROJECT_ROOT}/voices/Camila_Sodi.mp3",
        "prompt": """Todos venimos de un mismo campo fuente, de una misma gran energía, de un mismo Dios, de un mismo 
        universo, como le quieras llamar. Todos somos parte de eso. Nacemos y nos convertimos en esto por un ratito 
        muy chiquito, muy chiquitito, que creemos que es muy largo y se nos olvida que vamos a regresar a ese lugar 
        de donde venimos, que es lo que tú creas, adonde tú creas, pero inevitablemente vas a regresar."""
    }
    # "Cristina": {
    #     "temp": 0.82,
    #     "top_p": 0.91,
    #     "chunk": 807,
    #     "penalty": 1.07,
    #     "ref_path": f"{PROJECT_ROOT}/voices/Elevenlabs_Cristina_Campos_optimized.wav",
    #     "prompt": """El agua, la confianza y el miedo. Una lección poderosa y reveladora sobre la verdadera
    #     protección y el poder de la preparación. Considera la profunda enseñanza que subyace a la instrucción"""
    # },
    # "ROSA": {
    #     "temp": 0.82,
    #     "top_p": 0.91,
    #     "chunk": 807,
    #     "penalty": 1.07,
    #     "ref_path": f"{PROJECT_ROOT}/voices/Elevenlabs_Rosa_Estela_optimized.wav",
    #     "prompt": """El agua, la confianza y el miedo. Una lección poderosa y reveladora sobre la verdadera
    #     protección y el poder de la preparación. Considera la profunda enseñanza que subyace a la instrucción"""
    # },
    # "ALEJANDRO": {
    #     "temp": 0.84,
    #     "top_p": 0.91,
    #     "chunk": 785,
    #     "penalty": 1.07,
    #     "ref_path": f"{PROJECT_ROOT}/voices/ElevenLabs_Alejandro_optimized.mp3",
    #     "prompt": "La mente lo es todo. La causa mental. La causa de todo -absolutamente todo- es mental, es decir, "
    #               "la mente es la que produce o causa todo en la vida del individuo."
    # },
    # "ALEJANDRO_BALLESTEROS": {
    #     "temp": 0.84,
    #     "top_p": 0.91,
    #     "chunk": 785,
    #     "penalty": 1.07,
    #     "ref_path": f"{PROJECT_ROOT}/voices/Elevenlabs_Alejandro_Ballesteros_optimized.wav",
    #     "prompt": """El agua, la confianza y el miedo. Una lección poderosa y reveladora sobre la verdadera
    #     protección y el poder de la preparación. Considera la profunda enseñanza que subyace a la instrucción"""
    # },
    # "ENRIQUE": {
    #     "temp": 0.84,
    #     "top_p": 0.91,
    #     "chunk": 785,
    #     "penalty": 1.07,
    #     "ref_path": f"{PROJECT_ROOT}/voices/Elevenlabs_Enrique_Nieto_optimized.wav",
    #     "prompt": """El agua, la confianza y el miedo. Una lección poderosa y reveladora sobre la verdadera
    #     protección y el poder de la preparación. Considera la profunda enseñanza que subyace a la instrucción"""
    # }
}


# Detectar si estamos en Windows o Linux
is_windows = platform.system() == "Windows"

# En Windows forzamos False, en Kaggle (Linux) podemos usar True si queremos velocidad
should_compile = False if is_windows else True

class FishTotalLab:

    def __init__(self):
        self.device = "cuda"
        self.checkpoint_dir = PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini"
        self.precision = torch.half
        logger.info("🎯 INICIANDO MOTOR INTEGRADO (PRODUCCIÓN + HYPER-SEARCH)")
        self.engine = self._load_models()
        torch.cuda.empty_cache()
        gc.collect()

    def _load_models(self):
        llama_queue = launch_thread_safe_queue(checkpoint_path=self.checkpoint_dir,
                                               device=self.device,
                                               precision=self.precision,
                                               compile=should_compile)

        decoder_model = load_decoder_model(config_name="modded_dac_vq",
                                           checkpoint_path=self.checkpoint_dir / "codec.pth",
                                           device=self.device)

        return TTSInferenceEngine(llama_queue=llama_queue,
                                  decoder_model=decoder_model,
                                  precision=self.precision,
                                  compile=should_compile)

    # --- FUNCIÓN DE PRODUCCIÓN (RESTAURADA Y MEJORADA) ---
    def generate_production_batch(self, text_to_speak):
        logger.info(f"🎙️ PRODUCCIÓN: Generando audios de alta fidelidad.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_folder = PROJECT_ROOT / f"produccion_final_{timestamp}"
        out_folder.mkdir(parents=True,
                         exist_ok=True
                         )

        for name, params in VOICE_PRESETS.items():
            logger.trace(
                f"🚀 Generando voz final: {name} | T={params['temp']} | P={params['top_p']} | C={params['chunk']}")

            with open(params['ref_path'], "rb") as f:
                audio_bytes = f.read()

            with torch.inference_mode():
                vq_tokens = self.engine.encode_reference(audio_bytes,
                                                         enable_reference_audio=True
                                                         )

                req = ServeTTSRequest(
                    text=text_to_speak,
                    references=[
                        ServeReferenceAudio(
                            audio=audio_bytes,
                            tokens=vq_tokens.tolist(),
                            text=params['prompt']
                        )],
                    max_new_tokens=2048,
                    chunk_length=params['chunk'],
                    top_p=params['top_p'],
                    temperature=params['temp'],
                    repetition_penalty=params['penalty'],
                    format="wav"
                )

                results = self.engine.inference(req)
                audio_parts = []
                for res in results:
                    chunk = res.audio if hasattr(res, 'audio') else res
                    if isinstance(chunk, tuple):
                        audio_parts.extend([x for x in chunk if isinstance(x, np.ndarray)])
                    elif isinstance(chunk, np.ndarray):
                        audio_parts.append(chunk)

                if audio_parts:
                    final_path = out_folder / f"FINAL_{name}_{timestamp}.wav"
                    sf.write(str(final_path), np.concatenate(audio_parts), 44100)
                    logger.success(f"✅ ¡LISTO! {name} guardado en: {final_path}")

            torch.cuda.empty_cache()
            gc.collect()

            # --- NUEVA SECCIÓN: GENERACIÓN DE ZIP FINAL ---
            try:
                zip_filename = f"lote_produccion_{timestamp}"
                zip_path = PROJECT_ROOT / zip_filename
                shutil.make_archive(str(zip_path), 'zip', out_folder)
                logger.success(f"📦 ¡TODO COMPRIMIDO! Descarga el archivo: {zip_filename}.zip")
            except Exception as e:
                logger.error(f"❌ No se pudo crear el ZIP: {e}")

    # --- FUNCIÓN HYPER SEARCH (ANCLAJE DINÁMICO) ---
    def run_hyper_search(self, text, num_tests=15):
        logger.info(f"🧪 Iniciando laboratorio Multivoz para {len(VOICE_PRESETS)} voces.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for voice_name, base_params in VOICE_PRESETS.items():
            voice_folder = PROJECT_ROOT / f"hyper_{voice_name}_{timestamp}"
            voice_folder.mkdir(parents=True,
                               exist_ok=True
                               )

            with open(base_params['ref_path'], "rb") as f:
                audio_bytes = f.read()
            with torch.inference_mode():
                vq_tokens = self.engine.encode_reference(audio_bytes,
                                                         enable_reference_audio=True
                                                         )

            t_base = base_params['temp']
            p_base = base_params['top_p']

            for i in range(num_tests):
                progress = i / (num_tests - 1) if num_tests > 1 else 0
                curr_t = round(t_base + (0.05 * progress), 2)  # Exploración fina
                curr_p = round(min(p_base + (0.04 * progress), 0.96), 2)
                curr_chunk = int(base_params['chunk'] + (200 * progress))
                curr_pen = 1.07  # Mantener fluidez ganada

                logger.trace(f"🌀 [{voice_name} {i + 1}/{num_tests}] | T={curr_t} | P={curr_p} | C={curr_chunk}")

                req = ServeTTSRequest(
                    text=text,
                    references=[
                        ServeReferenceAudio(audio=audio_bytes, tokens=vq_tokens.tolist(), text=base_params['prompt'])],
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
                    clean_parts = []
                    for p in audio_parts:
                        if isinstance(p, tuple):
                            clean_parts.extend([x for x in p if isinstance(x, np.ndarray)])
                        elif isinstance(p, np.ndarray):
                            clean_parts.append(p)

                    if clean_parts:
                        sf.write(str(voice_folder / f"{voice_name}_TEST_{i + 1:02d}.wav"), np.concatenate(clean_parts),
                                 44100)
                except Exception as e:
                    logger.error(f"Error en {voice_name}: {e}")

            shutil.make_archive(str(PROJECT_ROOT / f"pruebas_{voice_name}_{timestamp}"), 'zip', voice_folder)
            logger.success(f"🏁 Pack de pruebas generado para {voice_name}")


if __name__ == "__main__":
    lab = FishTotalLab()

    TEXTO_PARA_PRODUCIR = """
        La mente lo es todo. 
        
        La causa mental.
        
        La causa de todo -absolutamente todo- es mental, es decir,  la mente es la que produce o causa todo en la vida del individuo.  
        Cuando reconozcamos, entendamos y aceptemos esta verdad,  habremos dado un paso muy importante en el progreso del desarrollo. 
        
        Si todo es mental, este es un universo mental, donde todo funciona por  medios mentales. Nosotros somos seres mentales, mentalidades buenas,  perfectas y eternas. 
        
        La mente sólo tiene una actividad, pensar. El pensamiento es todo lo  de la mente lo único que somos y tenemos es pensamiento, por ello, el  pensamiento es lo más importante de todo.
    """

    TEXTO_PARA_PRODUCIR_CUSTOM = """Todos venimos de un mismo campo fuente, de una misma gran energía, de un mismo 
    Dios, de un mismo universo, como le quieras llamar. Todos somos parte de eso. Nacemos y nos convertimos en esto 
    por un ratito muy chiquito, muy chiquitito, que creemos que es muy largo y se nos olvida que vamos a regresar a 
    ese lugar de donde venimos, que es lo que tú creas, adonde tú creas, pero inevitablemente vas a regresar.
    """

    # 1. Ejecutar producción final con los parámetros que ya te gustaron (T=0.82/0.84)
    lab.generate_production_batch(TEXTO_PARA_PRODUCIR_CUSTOM)

    # 2. (Opcional) Si quieres seguir probando aún más fluidez:
    # lab.run_hyper_search(TEXTO_PARA_PRODUCIR_CUSTOM, num_tests=15)