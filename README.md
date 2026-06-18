### Vaakya: Privacy first AR Captioning Glasses
**Python, Vosk STT, Raspberry Pi, Pygame**

---

Vaakya (Sanskrit: Speech) is a real time AR captioning system built for the 28.8 million working-age adults with hearing los in the US. This specific glasses was targeted for high stakes environments like clinical intake meetings, classrooms and legal proceedings. Existing solutions are too pricey or fail on latency or are not privacy comliant.

---

##Architecture
The main challenge with this was getting end to end latency. Our system was made in a way where the microphone captures the audio to display render under 300ms on a Raspberry Pi, which is below the perceptual threshold for caption lag during natural speech of humans. All the current existing STT pipelines are calling a cloud API (600ms+ round trip latency) or run too slowly on edge hardware to be usable in real conversations.

To solve this, I built the following:
- **Dual threaded inference pipleline** - an `stt_worker` thread which continously drains the audio queue and runs the Vosk ASR inference which is seperate from the Pygame render loop running at 60 FPS. This prevents audio processing from blocking the display and keeps caption updates smooth when during long sustained speech.
- -**Low latency audio capture** - tuned `AUDIO_BLOCKSIZE` to 1000 samples on the Raspberry Pi to minimize buffering delay between the mic input and inference without stopping the Vosk recongizer.
- - **Caption merge logic** - implemented a 'mayber_merge()' function which has a 600ms merge window. This is to prevent any type of mini sentence spam from Vosk's incremental final results.
  - -**Zero persistence audio architecture** - all audio is processed in RAM via an on device Vosk model. None of the data is written on the disk and none of the data is transmitted to remote servers and no conversation is retained after the session either.
  - -**HUD integrated display** -  All the captions render are centered on a Vufine headmount display. The text is wrap scaled to display width so the captions are never overflowing off the screen.

----
