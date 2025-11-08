import torch 
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import librosa
import io

MODEL_PATH="./models/"
BASE_MODEL_NAME="openai/whisper-small"

model = None
processor = None
forced_decoder_ids = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    global model, processor, forced_decoder_ids

    if model is None:
        print("🚀 Whisper 모델을 로드하는 중...")
        processor = WhisperProcessor.from_pretrained(BASE_MODEL_NAME,language="korean", task="transcribe")
        
        base_model = WhisperForConditionalGeneration.from_pretrained(
            BASE_MODEL_NAME
        )

        # 어댑터 병합 
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        model = model.to(DEVICE)
        model.eval()

        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="korean", task="transcribe"
        )

        print("✅ 모델 로드 완료.")

def transcribe_audio_file(audio_bytes:bytes) -> str:
    if model is None or processor is None:
        raise RuntimeError("모델이 로드되지 않았습니다.")
    
    audio_steam = io.BytesIO(audio_bytes)
    speech_array, _ = librosa.load(audio_steam, sr=16000, mono=True)

    input_features = processor(
        speech_array, 
        sample_rate=16000, 
        return_tensors="pt"
    ).input_features

    input_features = input_features.to(DEVICE)
    
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids
        )
        
    # 결과 디코딩
    transcription = processor.batch_decode(
        predicted_ids, 
        skip_special_tokens=True
    )[0]
    
    return transcription.strip()