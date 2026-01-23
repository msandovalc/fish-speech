import os
import sys
import torch
import gc
import shutil
import re
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger
from datetime import datetime
import platform

# --- SYSTEM CONFIGURATION ---
logger.remove()
logger.add(sys.stdout, colorize=True, level="TRACE",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

# Optimize Memory Fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- CONSTANTS ---
# Auto-detect project root
PROJECT_ROOT = Path(__file__).resolve().parent

# --- IMPORTS WITH FALLBACK ---
try:
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.dac.inference import load_model as load_decoder_model
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
    from fish_speech.utils import set_seed
except ImportError:
    # If running from a notebook cell, add root to path
    sys.path.insert(0, str(PROJECT_ROOT))
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.dac.inference import load_model as load_decoder_model
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
    from fish_speech.utils import set_seed

# --- VOICE PRESETS (OPTIMIZED FOR S1-MINI) ---
# NOTE: Temperatures lowered to ~0.75 and Penalty to ~1.05 to prevent metallic artifacts.
VOICE_PRESETS = {
    # "MARLENE": {
    #     "temp": 0.75,
    #     "top_p": 0.90,
    #     "chunk": 900,
    #     "penalty": 1.05,
    #     "ref_path": str(PROJECT_ROOT / "voices" / "ElevenLabs_Marlene.mp3"),
    #     "prompt": """La mente lo es todo. La causa mental. La causa de todo -absolutamente todo- es mental, es decir,
    #     la mente es la que produce o causa todo en la vida del individuo.
    #
    #     Cuando reconozcamos, entendamos y aceptemos esta verdad, habremos dado un paso muy importante en el progreso del desarrollo.
    #
    #     Si todo es mental, este es un universo mental, donde todo funciona por medios mentales. Nosotros somos seres
    #     mentales, mentalidades buenas, perfectas y eternas.
    #
    #     La mente sólo tiene una actividad, pensar. El pensamiento es todo lo de la mente lo único que somos y tenemos es
    #     pensamiento, por ello, el pensamiento es lo más importante de todo.
    #     """
    #     "style_tags": "(calm) (narrative)"
    # },
    # "MARGARITA": {
    #     "temp": 0.82,
    #     "top_p": 0.91,
    #     "chunk": 900,
    #     "penalty": 1.07,
    #     "ref_path": str(PROJECT_ROOT / "voices" / "Margarita_Navarrete.wav"),
    #     "prompt": """Mira te comparto, hicimos tres cuartos más y no suelta todavía el sistema y otros detallitos,
    #     pero mira lo que te quiero comentar es que sé que suena raro, sé que se requiere dinero para el intercambio
    #     de lo que se desea, sin embargo todo lo que decidas hacer, hazlo porque deseas hacerlo. Lo común es buscar
    #     hacerlo porque necesitas, y entonces si se empieza a hacer todo desde la necesidad, desde pues es que Magui
    #     si lo requiero para los pagos, quedó bien justito ahorita, entonces te me vas a empezar a estresar más. Haz
    #     las cosas porque te gusta lo que estás haciendo y de lo que te gusta empieza a hacer más, pero porque te gusta.
    #
    #     ¿Cómo voy a poder eliminar la carencia del gusto? Por eso son las líneas, a mí me pasó, te digo tiene poco
    #     que saque el crédito.
    #     """,
    #     "style_tags": "(calm) (deep voice)"
    # },
    "CAMILA": {
        "temp": 0.65,
        "top_p": 0.70,
        "chunk": 300,
        "penalty": 1.035, #1.035
        "ref_path": str(PROJECT_ROOT / "voices" / "Camila_Sodi.mp3"),
        "prompt": """Todos venimos de un mismo campo fuente, de una misma gran energía, de un mismo Dios, de un mismo 
        universo, como le quieras llamar. Todos somos parte de eso. Nacemos y nos convertimos en esto por un ratito 
        muy chiquito, muy chiquitito, que creemos que es muy largo y se nos olvida que vamos a regresar a ese lugar 
        de donde venimos, que es lo que tú creas, adonde tú creas, pero inevitablemente vas a regresar.""",
        "style_tags": "(calm) (narrator)"
    }
    # "CRISTINA": {
    #     "temp": 0.75,
    #     "top_p": 0.90,
    #     "chunk": 900,
    #     "penalty": 1.05,
    #     "ref_path": str(PROJECT_ROOT / "voices" / "Elevenlabs_Cristina_Campos.wav"),
    #     "prompt": "El agua, la confianza y el miedo...",
    #     "style_tags": "(calm) (narrative)"
    # },
    # "ROSA": {
    #     "temp": 0.75,
    #     "top_p": 0.90,
    #     "chunk": 900,
    #     "penalty": 1.05,
    #     "ref_path": str(PROJECT_ROOT / "voices" / "Elevenlabs_Rosa_Estela.wav"),
    #     "prompt": "El agua, la confianza y el miedo...",
    #     "style_tags": "(calm) (soft)"
    # },
    # "ALEJANDRO": {
    #     "temp": 0.75,
    #     "top_p": 0.85,
    #     "chunk": 900,
    #     "penalty": 1.10,
    #     "ref_path": str(PROJECT_ROOT / "voices" / "ElevenLabs_Alejandro.mp3"),
    #     "prompt": "(serious) (calm) La mente lo es todo.",
    #     "style_tags": "(serious) (calm)"
    # },
    # "ALEJANDRO_BALLESTEROS": {
    #     "temp": 0.75,
    #     "top_p": 0.90,
    #     "chunk": 900,
    #     "penalty": 1.05,
    #     "ref_path": str(PROJECT_ROOT / "voices" / "Elevenlabs_Alejandro_Ballesteros.wav"),
    #     "prompt": "El agua, la confianza y el miedo...",
    #     "style_tags": "(serious) (deep)"
    # },
    # "ENRIQUE": {
    #     "temp": 0.75,
    #     "top_p": 0.90,
    #     "chunk": 900,
    #     "penalty": 1.05,
    #     "ref_path": str(PROJECT_ROOT / "voices" / "Elevenlabs_Enrique_Nieto.wav"),
    #     "prompt": "El agua, la confianza y el miedo...",
    #     "style_tags": "(narrative) (serious)"
    # }
}

# Platform detection
is_windows = platform.system() == "Windows"
should_compile = False if is_windows else True


class FishTotalLab:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_dir = PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini"
        self.precision = torch.half
        self.vocal_dna_cache = {}
        logger.info(f"🎯 STARTING FISH LABORATORY | Device: {self.device} | Compile: {should_compile}")
        self.engine = self._load_models()
        torch.cuda.empty_cache()
        gc.collect()

    def _load_models(self):
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=self.checkpoint_dir,
            device=self.device,
            precision=self.precision,
            compile=should_compile
        )
        decoder_model = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=self.checkpoint_dir / "codec.pth",
            device=self.device
        )
        return TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=self.precision,
            compile=should_compile
        )

    def clean_text(self, text):
        if not text: return ""
        text = re.sub(r'([.!?…])(?=\S)', r'\1 ', text)
        text = text.replace("\n", " ").replace("\t", " ")
        return re.sub(r'\s+', ' ', text).strip()

    def split_text(self, text, max_chars=200):
        """
        Split en 200.
        Suficiente para una frase larga, pero seguro para la memoria.
        """
        text = self.clean_text(text)
        sentences = re.split(r'(?<=[.!?…])\s+', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if not sentence.strip(): continue
            if len(current_chunk) + len(sentence) < max_chars:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def _crossfade_chunks(self, audio_list, crossfade_ms=30, sample_rate=44100):
        if not audio_list: return None
        if len(audio_list) == 1: return audio_list[0]
        fade_samples = int(sample_rate * crossfade_ms / 1000)
        combined = audio_list[0]
        for next_chunk in audio_list[1:]:
            if len(combined) < fade_samples or len(next_chunk) < fade_samples:
                combined = np.concatenate((combined, next_chunk))
                continue
            fade_out = np.linspace(1, 0, fade_samples)
            fade_in = np.linspace(0, 1, fade_samples)
            tail = combined[-fade_samples:] * fade_out
            head = next_chunk[:fade_samples] * fade_in
            overlap = tail + head
            combined = np.concatenate((combined[:-fade_samples], overlap, next_chunk[fade_samples:]))
        return combined

    def _normalize_audio(self, audio_data, target_db=-1.0):
        max_val = np.abs(audio_data).max()
        if max_val == 0: return audio_data
        target_amp = 10 ** (target_db / 20)
        return audio_data * (target_amp / max_val)

    def generate_audio_for_params(self, voice_key, text, temp, top_p, penalty, chunk_size, style_tags,
                                  seed_base: int = 1234):
        if voice_key not in VOICE_PRESETS:
            logger.error(f"Voice {voice_key} not found.")
            return None

        params = VOICE_PRESETS[voice_key]
        set_seed(seed_base)

        cache_key = (voice_key, params["ref_path"])
        if cache_key in self.vocal_dna_cache:
            audio_bytes = self.vocal_dna_cache[cache_key]
        else:
            with open(params["ref_path"], "rb") as f:
                audio_bytes = f.read()
            self.vocal_dna_cache[cache_key] = audio_bytes

        # --- CHUNKS DE 200 ---
        text_chunks = self.split_text(text, max_chars=200)

        raw_parts = []
        hist_tokens = None
        hist_text = None

        # Obtenemos los tags del preset si no se pasaron explícitamente
        current_tags = style_tags if style_tags else params.get("style_tags", "")

        for i, chunk_text in enumerate(text_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text: continue

            # --- ESTRATEGIA: REINYECCIÓN CONSTANTE ---
            # Volvemos a poner los tags en CADA chunk.
            # Esto mantiene el ritmo "lento" y "calmado" a la fuerza.
            # processed_text = f"{current_tags} {chunk_text}"
            processed_text = f"{current_tags} {chunk_text}" if i == 0 else chunk_text

            # --- AUTO-RETRY MATEMÁTICO ---
            max_retries = 3
            best_attempt = None

            for attempt in range(max_retries):
                if attempt > 0:
                    set_seed(seed_base + i + attempt * 100)

                req = ServeTTSRequest(
                    text=processed_text,
                    references=[ServeReferenceAudio(audio=audio_bytes, text=params["prompt"])],
                    use_memory_cache="on",
                    chunk_length=chunk_size,
                    max_new_tokens=1024,
                    top_p=top_p,
                    temperature=temp,
                    repetition_penalty=penalty,
                    format="wav",
                    prompt_text=[hist_text] if hist_text is not None else None,
                    prompt_tokens=[hist_tokens] if hist_tokens is not None else None,
                )

                final_res = None
                for res in self.engine.inference(req):
                    if res.code == "final":
                        final_res = res
                        break

                # --- EL JUEZ IMPLACABLE ---
                if final_res and final_res.codes is not None:
                    num_tokens = final_res.codes.shape[1]

                    # Regla de Oro: En español normal, necesitas al menos 1 token por caracter
                    # (aunque usualmente es 1.3 - 1.5). Si es menos de 1.0, SE COMIÓ TEXTO.
                    # Ejemplo: Texto de 100 letras -> Mínimo 100 tokens.
                    min_tokens_needed = len(chunk_text)

                    if num_tokens < min_tokens_needed:
                        logger.warning(
                            f"⚠️ Chunk incompleto ({num_tokens} tokens vs {len(chunk_text)} letras). Reintentando...")
                        continue

                    best_attempt = final_res
                    break

            if best_attempt is None and final_res is not None:
                logger.error(f"❌ Falló reintento: '{chunk_text[:15]}...'. Usando último.")
                best_attempt = final_res

            if best_attempt is None or best_attempt.audio is None:
                continue

            sr, audio_np = best_attempt.audio
            raw_parts.append(audio_np)

            # --- MEMORIA BALANCEADA (50) ---
            # Mantenemos 50 para limpiar el "ruido" pero permitir la unión.
            if best_attempt.codes is not None:
                codes = torch.from_numpy(best_attempt.codes).to(torch.int)
                keep = 50
                if codes.shape[1] > keep:
                    codes = codes[:, -keep:]
                hist_tokens = codes
                hist_text = chunk_text

        if not raw_parts:
            return None

        merged = self._crossfade_chunks(raw_parts, crossfade_ms=30)
        final = self._normalize_audio(merged)
        return final

    def run_hyper_search(self, text, num_tests=5):
        logger.info(f"🧪 Starting Hyper-Search for {len(VOICE_PRESETS)} voices.")
        timestamp = datetime.now().strftime("%H%M%S")

        for voice_name, base_params in VOICE_PRESETS.items():
            voice_folder = PROJECT_ROOT / f"LAB_{voice_name}_{timestamp}"
            voice_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"🔬 Testing Voice: {voice_name}")

            for i in range(num_tests):
                curr_temp = base_params['temp']
                curr_pen = base_params['penalty']
                curr_chunk = base_params['chunk']

                logger.trace(f"🌀 Test {i + 1}: Chunk Size={curr_chunk} | (T={curr_temp}, P={curr_pen})")

                audio = self.generate_audio_for_params(
                    voice_name,
                    text,
                    temp=curr_temp,
                    top_p=base_params['top_p'],
                    penalty=curr_pen,
                    chunk_size=curr_chunk,
                    style_tags=base_params.get("style_tags", "")
                )

                if audio is not None:
                    filename = f"{voice_name}_FinalFixed_{timestamp}.wav"
                    sf.write(str(voice_folder / filename), audio, 44100, subtype="PCM_16")
                    logger.success(f"📦 Test pack created for {voice_name}__{filename}")

            shutil.make_archive(str(PROJECT_ROOT / f"RESULTS_{voice_name}_{timestamp}"), 'zip', voice_folder)
            logger.success(f"📦 Test pack created for {voice_name}")


if __name__ == "__main__":
    lab = FishTotalLab()

    # TEXTO DE PRUEBA
    TEST_TEXT = """
            Todos venimos de un mismo campo fuente, de una misma gran energía, de un mismo Dios, de un mismo
            universo, como le quieras llamar. Todos somos parte de eso. Nacemos y nos convertimos en esto por un ratito,
            muy chiquito, muy chiquitito, que creemos que es muy largo y se nos olvida que vamos a regresar a ese lugar
            de donde venimos.

            Escucha bien esto. No eres una gota en el océano, eres el océano entero en una gota. Tu imaginación no es un estado
            de fantasía o ilusión, es la verdadera realidad esperando ser reconocida. Cuando cierras los ojos y asumes el
            sentimiento de tu deseo cumplido, no estás "fingiendo", estás accediendo a la cuarta dimensión, al mundo de las
            causas, donde todo ya existe. Lo que ves afuera, en tu mundo físico, es simplemente una pantalla retrasada, un
            eco de lo que fuiste ayer, de lo que pensaste ayer.

            Si tu realidad actual no te gusta, deja de pelear con la pantalla. No puedes peinar tu reflejo en el espejo,
            tienes que peinarte tú. Debes cambiar la concepción que tienes de ti mismo. Pregúntate: ¿Quién soy yo ahora?
            Si la respuesta no es "Soy próspero", "Soy amado", "Soy saludable", entonces estás usando tu poder divino en tu
            contra. El universo no te juzga, simplemente te dice "SÍ". Si dices "estoy arruinado", el universo dice "SÍ, lo estás".
            Si dices "Soy abundante", el universo dice "SÍ, lo eres".

            Por lo tanto, el secreto no es el esfuerzo físico ni la lucha externa. El secreto es el cambio interno de estado.
            Moverte, en tu mente, del estado de carencia al estado de posesión. Sentir la textura de la realidad que deseas
            hasta que sea tan natural que ya no la busques, porque sabes que ya la tienes. Y cuando esa certeza interna hace
            clic, el mundo exterior no tiene más remedio que reorganizarse para reflejar tu nueva verdad. E inevitablemente,
            vas a regresar a tu poder.
        """

    lab.run_hyper_search(TEST_TEXT, num_tests=1)