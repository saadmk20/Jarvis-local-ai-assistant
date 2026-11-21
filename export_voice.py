from pathlib import Path
from pocket_tts import TTSModel, export_model_state

folder = Path(__file__).parent

#model = TTSModel.load_model()

model = TTSModel.load_model(language="french")

state = model.get_state_for_audio_prompt(str(folder / "jaravis_fr.wav"))
export_model_state(state, str(folder / "jarvis.fr.safetensors"))
print("Done.")