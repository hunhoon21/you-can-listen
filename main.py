import os
import subprocess
from dotenv import load_dotenv
import whisper
from openai import OpenAI
import srt
from datetime import timedelta
from pathlib import Path

# Load .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGUAGE = os.getenv("LANGUAGE", "Korean")
MODEL = os.getenv("MODEL", "medium")

AUDIO_DIR = Path("audio")
SUBS_DIR = Path("subs")
AUDIO_DIR.mkdir(exist_ok=True)
SUBS_DIR.mkdir(exist_ok=True)

def download_audio(youtube_url: str) -> str:
    filename = AUDIO_DIR / "audio.mp3"
    subprocess.run([
        "yt-dlp", "-f", "bestaudio", "--extract-audio",
        "--audio-format", "mp3", "-o", str(filename), youtube_url
    ], check=True)
    return str(filename)

def transcribe_audio(filepath: str, model_name: str):
    model = whisper.load_model(model_name)
    return model.transcribe(filepath)

def save_srt(segments, path):
    subtitles = [
        srt.Subtitle(index=i,
                     start=timedelta(seconds=seg["start"]),
                     end=timedelta(seconds=seg["end"]),
                     content=seg["text"])
        for i, seg in enumerate(segments)
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subtitles))

def translate_segments(segments, batch_size=9):
    client = OpenAI(api_key=OPENAI_API_KEY)

    translated = []
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i+batch_size]
        original_lines = [s["text"] for s in batch]

        n_lines = len(original_lines)

        prompt = (
            "다음은 Whisper로 추출된 영어 자막입니다. 이 자막은 문장 단위가 아닌, 일정한 길이로 잘려 있어서 중간에 끊긴 문장들도 포함되어 있습니다.\n\n"
            "당신의 작업:\n"
            "- 각 줄을 자연스럽고 일관된 존댓말로 번역해 주세요.\n"
            "- 문맥상 두 줄이 하나의 문장이 되더라도, 반드시 원문 줄 수에 맞춰 번역 결과도 같은 줄 수로 출력해 주세요.\n"
            "- 의미상 두 줄이 한 문장이라면, 해당 번역 문장을 그 두 줄에 동일하게 반복해서 출력해 주세요.\n\n"
            "출력 형식:\n"
            f"- 이번 입력에서는 {n_lines}줄이 주어질거고, 반드시 {n_lines}줄의 한국어 번역 결과를 줄바꿈 기준으로 출력해야 합니다.\n"
            "- 원문 줄과 번역 줄의 수가 정확히 일치해야 합니다. 명심하세요.\n\n"
            "예시:\n"
            "1. His company Scale AI supplies high quality training data to Nvidia, OpenAI, General Motors,\n"
            "2. Microsoft, and Meta.\n"
            "→\n"
            "1. 그의 회사인 Scale AI는 Nvidia, OpenAI, 제너럴 모터스, 마이크로소프트, 그리고 메타에 고품질 훈련 데이터를 제공합니다.\n"
            "2. 그의 회사인 Scale AI는 Nvidia, OpenAI, 제너럴 모터스, 마이크로소프트, 그리고 메타에 고품질 훈련 데이터를 제공합니다.\n\n"
            "아래는 번역할 원문입니다:\n\n" +
            "\n".join(original_lines)
        )
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": prompt,
            }]
        )
        translated_lines = res.choices[0].message.content.strip().split("\n")
        translated_lines = [line[3:] for line in translated_lines if line.strip()]

        if len(translated_lines) != len(original_lines):
            print(f"{len(translated_lines)=} {len(original_lines)=}")
            for i, line in enumerate(original_lines):
                print(f"원문 {i+1}: {line}")
            for i, line in enumerate(translated_lines):
                print(f"번역 {i+1}: {line}")
            raise ValueError("번역된 줄 수가 원문과 다릅니다")

        for seg, trans in zip(batch, translated_lines):
            translated.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": trans.strip()
            })
    return translated

def main(youtube_url):
    audio_path = download_audio(youtube_url)
    result = transcribe_audio(audio_path, MODEL)

    # Save original SRT
    save_srt(result["segments"], SUBS_DIR / "original.srt")

    # Translate and save translated SRT
    translated = translate_segments(result["segments"])
    save_srt(translated, SUBS_DIR / "translated.srt")

    print("✅ 자막 생성 및 번역 완료!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: poetry run python main.py [YOUTUBE_URL]")
    else:
        main(sys.argv[1])
