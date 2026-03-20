from faster_whisper import WhisperModel
import tempfile

model = WhisperModel("base", device="cpu")

async def transcribe_video(video):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp:

        content = await video.read()
        temp.write(content)
        temp.flush()

        segments, _ = model.transcribe(temp.name)

        text = ""

        for seg in segments:
            text += seg.text + " "

    return text.strip()
