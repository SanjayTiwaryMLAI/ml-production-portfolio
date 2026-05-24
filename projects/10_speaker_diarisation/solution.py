"""
Use Case: Speaker Diarisation Deployment — Async Inference Architecture
------------------------------------------------------------------------
Problem : Real-time deployment failed due to memory/latency constraints.
Approach: Async queue → HuggingFace diarisation → Speaker summary.
"""
import time, random, string, numpy as np
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Optional


@dataclass
class Segment:
    speaker: str
    start:   float
    end:     float

    @property
    def duration(self): return round(self.end - self.start, 2)

    def to_dict(self): return {"speaker":self.speaker,"start":self.start,"end":self.end,"duration":self.duration}


class Diariser:
    def __init__(self, token=None):
        self.pipeline = None
        try:
            from pyannote.audio import Pipeline
            self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
            print("pyannote loaded")
        except Exception:
            print("Simulation mode (pyannote not available)")

    def diarise(self, audio_path):
        if self.pipeline:
            return [Segment(spk, t.start, t.end).to_dict()
                    for t, _, spk in self.pipeline(audio_path).itertracks(yield_label=True)]
        # Simulate
        rng = np.random.default_rng(42)
        n   = rng.integers(2, 5)
        spks = [f"SPEAKER_{i:02d}" for i in range(n)]
        segs, t = [], 0.0
        for _ in range(20):
            dur = round(rng.uniform(2, 15), 2)
            segs.append(Segment(rng.choice(spks), t, t+dur).to_dict())
            t += dur + round(rng.uniform(0, 1), 2)
        return segs


class AsyncQueue:
    def __init__(self):
        self.jobs = {}
        self.q    = Queue()
        Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        from time import sleep
        diariser = Diariser()
        while True:
            jid = self.q.get()
            self.jobs[jid]["status"] = "PROCESSING"
            self.jobs[jid]["result"] = diariser.diarise(self.jobs[jid]["path"])
            self.jobs[jid]["status"] = "COMPLETED"
            self.q.task_done()

    def submit(self, path):
        jid = "D-" + "".join(random.choices(string.digits, k=6))
        self.jobs[jid] = {"status":"QUEUED","path":path,"result":None}
        self.q.put(jid)
        return jid

    def wait(self, jid, timeout=30):
        start = time.time()
        while time.time()-start < timeout:
            if self.jobs[jid]["status"] == "COMPLETED":
                return self.jobs[jid]
            time.sleep(0.3)
        return {"status":"TIMEOUT"}


def speaker_summary(segs):
    from collections import defaultdict
    tot, cnt = defaultdict(float), defaultdict(int)
    for s in segs:
        tot[s["speaker"]] += s["duration"]
        cnt[s["speaker"]] += 1
    total = sum(tot.values())
    return {spk: {"time_s": round(t,2), "share_pct": round(t/total*100,1), "segments": cnt[spk]}
            for spk, t in sorted(tot.items(), key=lambda x: -x[1])}


if __name__ == "__main__":
    queue = AsyncQueue()
    time.sleep(0.5)  # Let worker start

    jid = queue.submit("meeting_audio.wav")
    print(f"Submitted: {jid}")
    result = queue.wait(jid)

    segs = result["result"]
    print(f"Segments: {len(segs)}")
    summary = speaker_summary(segs)
    print("\nSpeaker Summary:")
    for spk, s in summary.items():
        print(f"  {spk}: {s['time_s']}s ({s['share_pct']}%) | {s['segments']} segments")
