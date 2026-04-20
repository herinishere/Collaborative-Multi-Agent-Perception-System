import cv2
import time
import numpy as np
from PIL import Image
from typing import TypedDict, List
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from groq import Groq
import threading
from collections import deque
from langgraph.graph import StateGraph, END

LLM_MODEL = "llama-3.1-8b-instant" 
client = Groq(api_key="YOUR_GROQ_API_KEY")

model = YOLO("yolov8m.pt") 
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model2 = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

class TemporalMemory:
    def __init__(self,maxlen=250):
        self._lock=threading.Lock()
        self._buffer=deque(maxlen=maxlen)
    
    def add(self,entry):
        with self._lock:
            entry["timestamp"]=time.time()
            self._buffer.append(entry)

    def scene_history(self,n):
        with self._lock:        
            scenes=[]
            for entry in list(self._buffer)[-n:]:
                if entry.get("scene"):                  
                    scenes.append(entry["scene"])
            return scenes
        
    def dominant_objects(self,n,min_frames):
        with self._lock:
            trend={}
            for entry in list(self._buffer)[-n:]:
                for obj in set(entry.get("objects",[])):
                    trend[obj]=trend.get(obj,0)+1
            return [k for k, v in sorted(trend.items(), key=lambda x: -x[1]) if v >= min_frames]

temporal_mem = TemporalMemory(maxlen=250)

class BaseMemory(TypedDict):
    frame: np.ndarray
    objects: List[str]
    object_counts: dict
    scene: str
    final_explanation: str
    is_consistent: bool
    critic_notes: str
    fps: float
    retry_count: int

snapshot = {
    "objects": [], "object_counts": {}, "final_explanation": "",
    "scene": "", "is_consistent": True, "critic_notes": "Initializing", "fps": 0.0,
    "memory_summary": "Waiting"
}

snapshot_lock = threading.Lock()

_pipeline_lock = threading.Lock()

class MemoryAgent:
    def __init__(self,interval):
        self.interval=interval
        self._stop=threading.Event()
        self._thread=threading.Thread(target=self._loop,daemon=True)
    
    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            self.run()
            self._stop.wait(self.interval)

    def run(self):
        with snapshot_lock:
            current_obs = set(snapshot["objects"])
        dominant = set(temporal_mem.dominant_objects(n=100, min_frames=40))
        recent_sc=temporal_mem.scene_history(n=50)
        new_arrivals=current_obs-dominant
        disappeared=dominant-current_obs
        parts=[]
        if current_obs:
            parts.append(f"Persistent: {', '.join(sorted(dominant if dominant else current_obs))}")
        if new_arrivals:
            parts.append(f"Just appeared: {', '.join(sorted(new_arrivals))}")
        if disappeared:
            parts.append(f"Left scene: {', '.join(sorted(disappeared))}")
        if len(recent_sc)>=2 and recent_sc[-1]!=recent_sc[0]:
            parts.append(f'Scene shift -> "{recent_sc[-1]}"')

        summary="|".join(parts) if parts else "Stable scene"
        with snapshot_lock:
            snapshot["memory_summary"]=summary

def VisionAgent(state: BaseMemory):
    results = model(state["frame"], imgsz=320, verbose=False)
    objs, counts = [], {}
    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls)]
            objs.append(label)
            counts[label] = counts.get(label, 0) + 1
    temporal_mem.add({"objects": objs, "object_counts": counts, "scene": snapshot.get("scene", "")})
    return {"objects": objs, "object_counts": counts}

def ContextAgent(state: BaseMemory):
    raw_img = Image.fromarray(cv2.cvtColor(state["frame"], cv2.COLOR_BGR2RGB))
    inputs = processor(images=raw_img, return_tensors="pt")
    out = model2.generate(**inputs, max_new_tokens=20)
    caption = processor.decode(out[0], skip_special_tokens=True).capitalize()
    return {"scene": caption}

def LanguageAgent(state: BaseMemory):
    scene = state.get("scene", "")
    objects = state.get("objects", [])

    prompt = f"""You are a real-time scene narrator for a vision system.

Detected objects: {objects}
Scene caption: "{scene}"

In ONE short sentence (max 15 words), describe what is happening in the scene naturally.
Do not use bullet points. Just one plain sentence. No preamble.
"""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=40,
        temperature=0.4,
    )
    llm_text = resp.choices[0].message.content.strip()
    llm_text = " ".join(llm_text.splitlines()).strip()
    return {"final_explanation": llm_text}

def CriticAgent(state: BaseMemory):
    prompt = f"""You are a consistency checker for a real-time vision system.

Detected objects: {state['objects']}
Scene description: "{state['scene']}"

Answer these two questions:

1. Does the description accurately reflect the detected objects? (yes/no)

2. Is there any contradiction between objects and description? (one sentence)

Output format exactly:

CONSISTENT: yes

NOTES: <one sentence — either confirm accuracy or describe the specific contradiction>"""
    resp = client.chat.completions.create(
        model=LLM_MODEL, 
        messages=[{"role": "user", "content": prompt}], 
        max_tokens=60, 
        temperature=0.1
        )
    raw = resp.choices[0].message.content.strip()
    consistent, notes = True, "Accurate."
    for line in raw.splitlines():
        if "CONSISTENT:" in line: consistent = "yes" in line.lower()
        elif "NOTES:" in line: notes = line.split(":", 1)[1].strip()
    retry_count = state.get("retry_count", 0)
    return {
        "is_consistent": consistent,
        "critic_notes": notes,
        "retry_count": retry_count
    }

def after_critic(state: BaseMemory)->str:
    if not state.get("is_consistent", True) and state.get("retry_count", 0)<2:
        state["retry_count"]=state.get("retry_count",0)+1
        return "retry"
    return "done"

workflow = StateGraph(BaseMemory)
workflow.add_node("vision", VisionAgent)
workflow.add_node("context", ContextAgent)
workflow.add_node("language", LanguageAgent)
workflow.add_node("critic", CriticAgent)
workflow.set_entry_point("vision")
workflow.add_edge("vision", "context")
workflow.add_edge("context", "language")
workflow.add_edge("language", "critic")
workflow.add_conditional_edges(
    "critic",
    after_critic,
    {"retry": "language","done": END}
)
app = workflow.compile()

def put_bg(frame, text, x, y, fs=0.6, color=(255, 255, 255)):
    (tw, fh), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    cv2.rectangle(frame, (x-5, y-fh-10), (x+tw+5, y+10), (0,0,0), -1)
    cv2.putText(frame, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)

def draw_overlay(frame):
    h,w=frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 130), (0, 0, 0), -1)
    accent=(0,255,0) if snapshot["is_consistent"] else (0,0,255)
    cv2.rectangle(frame, (0, 0), (w, 5), accent, -1)
    put_bg(frame, f"FPS: {snapshot['fps']:.1f}", 20, 25, fs=0.7)
    put_bg(frame, f"CONTEXT: {snapshot['scene']}", 20, 45, fs=0.5, color=(100, 220, 255))
    put_bg(frame, f"LLM: {snapshot['final_explanation']}", 20, 75, fs=0.5, color=(200, 200, 200)) 
    put_bg(frame, f"MEM: {snapshot['memory_summary']}", 20, 130, fs=0.5, color=(255, 220, 80))
    put_bg(frame, f"CRITIC: {snapshot['critic_notes']}", 20, 110, fs=0.5, color=accent)
    if snapshot["objects"]:
        names = list(set(snapshot["objects"]))
        names.sort()
        parts = []
        for name in names[:5]:
            count = snapshot["object_counts"].get(name, 1)
            display_text = name.upper() + " x" + str(count)
            parts.append(display_text)
        summary = " | ".join(parts)
        put_bg(frame, "DET: " + summary, 20, h - 30, fs=0.5, color=(50, 255, 50))

def run_system():
    print("1→Webcam | 2→Image | 3→Video")
    choice = input("Choice: ")
    win_name = "Perception System"
    cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)

    if choice=='2':
        path=input("Image path: ").strip().replace("'","").replace('"','')
        frame=cv2.imread(path)
        frame=cv2.resize(frame,(1280,720))
        result=app.invoke({"frame": frame})
        with snapshot_lock:
            snapshot.update(result)
        draw_overlay(frame)
        cv2.imshow(win_name, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    
    source=0 if choice=='1' else input("Video Path: ").strip().replace("'","").replace('"','')
    cap=cv2.VideoCapture(source)
    mem_agent=MemoryAgent(interval=3.0)
    mem_agent.start()
    frame_id=0
    fps=0.0
    fps_t=time.time()
    while True:
        ret,frame=cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if frame_id%15==0:
            fps=10/(time.time()-fps_t)
            fps_t=time.time()
            with snapshot_lock:
                snapshot["fps"] = fps
        if frame_id%30==0:
            if _pipeline_lock.acquire(blocking=False):
                def _task(f=frame.copy()):
                    try:
                        res=app.invoke({"frame":f})
                        with snapshot_lock:
                            snapshot.update(res)
                    finally:
                        _pipeline_lock.release()
                threading.Thread(target=_task, daemon=True).start()
        draw_overlay(frame)
        cv2.imshow(win_name, frame)
        frame_id += 1
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    mem_agent.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_system()
