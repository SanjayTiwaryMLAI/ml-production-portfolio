"""
Use Case: Scalable Speech-to-Text Infrastructure using Whisper
--------------------------------------------------------------
Problem : Large-scale audio transcription needs async, scalable infra.
Approach: Async endpoint → Whisper model → Batch pipeline.
"""
import time, random, string
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Optional


@dataclass
class Job:
    job_id:     str
    audio_path: str
    language:   Optional[str] = None
    status:     str = "QUEUED"
    transcript: Optional[str] = None
    elapsed_s:  Optional[float] = None


SAMPLE_TRANSCRIPTS = [
    "Quarterly revenue exceeded targets by fifteen percent this year.",
    "This lecture covers the fundamentals of machine learning and neural networks.",
    "The meeting has been rescheduled to Tuesday at three PM.",
    "Please review the attached documents before the board meeting.",
    "The new product launch is scheduled for the first quarter.",
]


class WhisperTranscriber:
    def __init__(self, model_size="base"):
        self.model = None
        try:
            import whisper, torch
            dev        = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = whisper.load_model(model_size, device=dev)
            print(f"Whisper {model_size} loaded on {dev}")
        except ImportError:
            print("Simulation mode")

    def transcribe(self, audio_path, language=None):
        t = time.time()
        if self.model:
            opts   = {"language": language} if language else {}
            result = self.model.transcribe(audio_path, **opts)
            return result["text"], round(time.time()-t, 2)
        time.sleep(0.3)
        return random.choice(SAMPLE_TRANSCRIPTS), round(time.time()-t, 2)


class AsyncPipeline:
    def __init__(self, n_workers=2):
        self.jobs      = {}
        self.queue     = Queue()
        transcriber    = WhisperTranscriber()
        for _ in range(n_workers):
            Thread(target=self._worker, args=(transcriber,), daemon=True).start()

    def _worker(self, transcriber):
        while True:
            jid  = self.queue.get()
            job  = self.jobs[jid]
            job.status = "PROCESSING"
            try:
                text, elapsed = transcriber.transcribe(job.audio_path, job.language)
                job.transcript = text
                job.elapsed_s  = elapsed
                job.status     = "COMPLETED"
            except Exception as e:
                job.status = f"FAILED: {e}"
            self.queue.task_done()

    def submit(self, audio_path, language=None):
        jid = "STT-" + "".join(random.choices(string.digits, k=6))
        self.jobs[jid] = Job(jid, audio_path, language)
        self.queue.put(jid)
        return jid

    def wait(self, jid, timeout=60):
        start = time.time()
        while time.time()-start < timeout:
            j = self.jobs[jid]
            if j.status in ("COMPLETED",) or j.status.startswith("FAILED"):
                return j
            time.sleep(0.2)
        return self.jobs[jid]

    def batch(self, files):
        ids = [self.submit(f) for f in files]
        return [self.wait(jid) for jid in ids]


if __name__ == "__main__":
    pipeline = AsyncPipeline(n_workers=2)
    time.sleep(0.5)

    files   = [f"audio_{i:03d}.wav" for i in range(5)]
    results = pipeline.batch(files)

    print("\nTranscription Results:")
    for j in results:
        icon = "✅" if j.status=="COMPLETED" else "❌"
        print(f"  {icon} [{j.job_id}] {j.elapsed_s}s")
        if j.transcript:
            print(f"     {j.transcript[:80]}...")
