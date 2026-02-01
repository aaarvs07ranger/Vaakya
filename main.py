import json
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass

import sounddevice as sd
import pygame
from vosk import Model, KaldiRecognizer

# =========================
# Config
# =========================
MODEL_PATH = "vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000

# Latency knobs (start here)
AUDIO_BLOCKSIZE = 1000          # lower -> lower latency; try 1000-2000 on Mac, 2000-4000 on Pi
UI_FPS = 60                     # render loop frequency

MAX_LINES = 3
LINE_TTL_SEC = 8.0              # expire old captions after this many seconds

# Merge behavior (reduces spammy micro-sentences)
MERGE_WINDOW_SEC = 0.6
MERGE_MAX_CHARS = 36

# Display settings
FULLSCREEN = False              # True on HDMI/Vufine when using on Raspberry Pi
FONT_SIZE = 80                  # larger
PADDING = 36
LINE_SPACING = 16
TEXT_COLOR = (255, 255, 255)
BG_COLOR = (0, 0, 0)
STATUS_COLOR = (180, 180, 180)

# =========================
# Data structures
# =========================
@dataclass
class CaptionLine:
    text: str
    t: float  # timestamp (monotonic seconds)

# =========================
# Queues and stop flag
# =========================
audio_queue = queue.Queue()
sentence_queue = queue.Queue()
stop_event = threading.Event()

# =========================
# Helpers
# =========================
def capitalize_first(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    return s[0].upper() + s[1:]

def wrap_text(font: pygame.font.Font, text: str, max_width: int) -> list:
    """
    Wrap text to fit within max_width (in pixels), using font metrics.
    Returns a list of lines.
    """
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]
    for w in words[1:]:
        test = current + " " + w
        if font.size(test)[0] <= max_width:
            current = test
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return lines

def audio_callback(indata, frames, time_info, status):
    audio_queue.put(bytes(indata))

# =========================
# Producer: Mic -> Vosk -> final captions queue
# =========================
def stt_worker():
    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(False)

    last_final = ""
    while not stop_event.is_set():
        try:
            data = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "").strip()
            if text and text != last_final:
                last_final = text
                sentence_queue.put(capitalize_first(text))

# =========================
# Consumer/UI: drain queue -> rolling buffer -> render
# =========================
def maybe_merge(lines: deque, new_text: str, now: float) -> None:
    if not lines:
        lines.append(CaptionLine(new_text, now))
        return

    last = lines[-1]
    if (now - last.t) <= MERGE_WINDOW_SEC and len(last.text) <= MERGE_MAX_CHARS:
        merged = (last.text.rstrip() + " " + new_text.lstrip()).strip()
        lines[-1] = CaptionLine(merged, now)
    else:
        lines.append(CaptionLine(new_text, now))

def expire_old(lines: deque, now: float) -> None:
    while lines and (now - lines[0].t) > LINE_TTL_SEC:
        lines.popleft()

def run_display():
    pygame.init()
    pygame.display.set_caption("Live Captions (Final-only)")

    flags = pygame.FULLSCREEN if FULLSCREEN else 0
    screen = pygame.display.set_mode((0, 0), flags)
    w, h = screen.get_size()

    font = pygame.font.SysFont(None, FONT_SIZE)
    status_font = pygame.font.SysFont(None, 28)

    clock = pygame.time.Clock()

    lines = deque(maxlen=MAX_LINES)
    paused = False

    def build_render_lines() -> list:
        """
        Converts the rolling buffer (caption sentences) into wrapped lines for rendering.
        Returns list of strings (already wrapped).
        """
        max_text_width = w - 2 * PADDING

        wrapped_all = []
        for item in list(lines):
            wrapped = wrap_text(font, item.text, max_text_width)
            wrapped_all.extend(wrapped)

        # If wrapping expands too much, keep the last lines that fit best visually
        # (keeps newest content)
        # We'll trim from the top if needed based on height.
        max_height = h - 2 * PADDING - 40  # leave room for status
        line_height = font.get_height() + LINE_SPACING
        max_lines_fit = max(1, max_height // line_height)
        if len(wrapped_all) > max_lines_fit:
            wrapped_all = wrapped_all[-max_lines_fit:]
        return wrapped_all

    def render():
        screen.fill(BG_COLOR)

        render_lines = build_render_lines()

        # Center vertically as a block
        line_height = font.get_height() + LINE_SPACING
        block_height = len(render_lines) * line_height - LINE_SPACING
        start_y = (h - block_height) // 2

        # Draw each line centered horizontally
        y = start_y
        for tline in render_lines:
            surf = font.render(tline, True, TEXT_COLOR)
            x = (w - surf.get_width()) // 2
            screen.blit(surf, (x, y))
            y += line_height

        # Status at bottom-left (small)
        status = "[PAUSED]" if paused else "[LIVE]"
        status_surf = status_font.render(status, True, STATUS_COLOR)
        screen.blit(status_surf, (PADDING, h - 40))

        pygame.display.flip()

    try:
        while True:
            now = time.monotonic()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    if event.key == pygame.K_SPACE:
                        paused = not paused

            if not paused:
                # Drain all available sentences immediately (stay caught up)
                while True:
                    try:
                        text = sentence_queue.get_nowait()
                        maybe_merge(lines, text, now)
                    except queue.Empty:
                        break

                expire_old(lines, now)

            render()
            clock.tick(UI_FPS)

    finally:
        pygame.quit()


# main

def main():
    print("Starting mic stream + STT worker...")
    stt_thread = threading.Thread(target=stt_worker, daemon=True)
    stt_thread.start()

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=AUDIO_BLOCKSIZE,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        print("Running display. Press ESC to quit, SPACE to pause.")
        run_display()

    stop_event.set()

if __name__ == "__main__":
    main()