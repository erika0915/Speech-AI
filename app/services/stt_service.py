import torch 
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import librosa
import io
from pydub import AudioSegment
import uuid
import os

MODEL_PATH="./models/"
BASE_MODEL_NAME="openai/whisper-small"

model = None
processor = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    global model, processor

    if model is None:
        print("🚀 Whisper 모델을 로드하는 중...")
        processor = WhisperProcessor.from_pretrained(BASE_MODEL_NAME)
        
        base_model = WhisperForConditionalGeneration.from_pretrained(
            BASE_MODEL_NAME
        )
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        model = model.to(DEVICE)
        model.eval()
        print("✅ 모델 로드 완료.")

def transcribe_audio_file(audio_bytes:bytes) -> str:
    
    filename = ""  # 디버깅 파일명을 위한 변수
    
    # --- 디버깅: 파일 크기 확인 및 저장 ---
    try:
        file_size = len(audio_bytes)
        print(f"DEBUG: 수신한 audio_bytes 크기: {file_size} 바이트")

        if file_size < 1000: # 1KB 미만 파일 거부
            print(f"❌ DEBUG: 파일 크기가 1KB 미만입니다. 처리를 중단합니다.")
            raise RuntimeError(f"파일이 너무 작음 (크기: {file_size})")

        filename = f"/app/debug_{uuid.uuid4().hex}.webm" 
        with open(filename, "wb") as f:
            f.write(audio_bytes)
        print(f"✅ DEBUG: 파일 저장 성공: {filename} (컨테이너 내부 경로)")
        
    except Exception as e:
        print(f"⚠️ DEBUG: 파일 저장/확인 중 에러 (STT 처리는 계속 시도): {e}")
    # --- 디버깅 코드 끝 ---

    if model is None or processor is None:
        raise RuntimeError("모델이 로드되지 않았습니다.")
    
    # --- pydub 변환 (필수) ---
    try:
        audio_stream = io.BytesIO(audio_bytes)
        audio_stream.seek(0) 
        audio_segment = AudioSegment.from_file(audio_stream)
        wav_stream = io.BytesIO()
        audio_segment.export(wav_stream, format="wav")
        wav_stream.seek(0)
    except Exception as e:
        print(f"오디오 변환 중 에러 발생 (pydub/ffmpeg): {e}")
        try:
            wav_stream = io.BytesIO(audio_bytes)
            wav_stream.seek(0)
        except Exception as inner_e:
             raise RuntimeError(f"오디오 처리 완전 실패: {inner_e}")
    
    # --- librosa 로드 ---
    speech_array, sampling_rate = librosa.load(wav_stream, sr=16000, mono=True)

    # --- processor 호출 (padding=True는 그대로 둡니다) ---
    processed_input = processor(
        speech_array, 
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True  # (이 옵션은 여러 파일을 처리할 때를 위해 둡니다)
    )
    
    input_features = processed_input.input_features.to(DEVICE)
    
    # --- ★★★★★ 진짜 최종 수정 지점 ★★★★★ ---
    # 'attention_mask'가 processed_input 딕셔너리에 '존재하는 경우'에만 가져오고,
    # '존재하지 않으면' (패딩이 안 됐으면) None을 사용합니다.
    if "attention_mask" in processed_input:
        attention_mask = processed_input.attention_mask.to(DEVICE)
    else:
        attention_mask = None # '주의력 지도'가 필요 없음
    # --- ★★★★★ 수정 끝 ★★★★★ ---


    with torch.no_grad():
        # --- 최신 방식으로 STT 실행 ---
        predicted_ids = model.generate(
            input_features,
            attention_mask=attention_mask, # None이 전달되어도 괜찮음
            language="korean",
            task="transcribe"
        )
        
    # --- 결과 디코딩 ---
    transcription = processor.batch_decode(
        predicted_ids, 
        skip_special_tokens=True
    )[0]
    
    # --- 디버깅용 임시 파일 삭제 ---
    if filename and os.path.exists(filename):
        try:
            os.remove(filename)
            print(f"✅ DEBUG: 임시 파일 삭제 성공: {filename}")
        except Exception as e:
            print(f"⚠️ DEBUG: 임시 파일 삭제 실패: {e}")

    return transcription.strip()