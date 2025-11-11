from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
from .services import stt_service
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    #  서버 시작 시 모델 로드 함수 호출 
    print("서버 시작 : Whisper 모델을 로드합니다...")
    stt_service.load_model()
    yield
    #  서버 종료 시 정리 
    print("서버가 종료되었습니다.")

app = FastAPI(lifespan=lifespan)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Whisper ASR API 서버가 실행 중입니다."}

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    transcription = stt_service.transcribe_audio_file(audio_bytes)
    print(f"Transcription: {transcription}")
    return {"transcription": transcription}