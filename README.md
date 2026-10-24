# Scouter 2

A camera that tells you who and what it sees, out loud.

Point a webcam (or a Raspberry Pi camera) at the room: Scouter recognises the people you've taught it, spots 80 kinds of everyday objects, and says what's there: *"Hello Hedi."*, *"I see a laptop and two cups."*

This is a rewrite of my 2020 Raspberry Pi project [Scouter](https://github.com/heditrigui/Scouter), which never got as far as actually speaking what it saw.

## Features

- **Face recognition** with [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (detection) and [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) (recognition). Teach it a person in a few seconds from the webcam, or from a folder of photos.
- **Object detection** with [YOLOX](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox): people, cups, phones, laptops, chairs, dogs… all 80 COCO classes.
- **Speech that doesn't nag**:
  - a known person is greeted when they show up, then not again for 30 seconds
  - an unknown face has to stay in view for 1.5 seconds before it's mentioned
  - an object has to be seen twice in a row before it's named, then not again for a minute
  - press **Space** any time to hear everything in view
- **Scouter-style overlay** with corner brackets, names, confidence and a status bar
- Runs on **Windows, macOS, Linux and Raspberry Pi**, with a webcam, a phone camera stream (`http://…/video`), or the Pi camera module, with or without a screen
- Fully offline once the models are downloaded. Faces are stored only on your machine, in `data/faces.npz`

## Install

Needs Python 3.10 or newer.

```bash
git clone https://github.com/heditrigui/scouter-2.git
cd scouter-2
python -m venv .venv
.venv/Scripts/activate        # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The three models (about 75 MB) download automatically on first run from the [OpenCV model zoo](https://github.com/opencv/opencv_zoo) and are checked against their SHA-256. To fetch them ahead of time: `python -m scouter download`.

## Use

```bash
python -m scouter enroll Hedi          # look at the webcam for a few seconds
python -m scouter enroll Sam --photos path/to/sam_photos
python -m scouter                      # start
python -m scouter people               # who it knows
python -m scouter forget Sam
```

While it runs: **Space** say what you see · **m** mute · **f** faces on/off · **o** objects on/off · **q** quit

Useful options for `python -m scouter`:

| Option | |
|---|---|
| `--camera 1` | another webcam |
| `--camera http://192.168.1.13:8080/video` | a phone running an IP-camera app |
| `--camera picamera` | the Raspberry Pi camera module |
| `--headless` | no window, for a Pi without a screen (Ctrl+C to stop) |
| `--quiet` | print what it would say instead of speaking |
| `--no-objects` / `--no-faces` | only faces / only objects |
| `--threshold 0.45` | stricter face matching (default 0.363) |
| `--cooldown 60` | seconds before repeating a person (objects wait twice as long) |

## Raspberry Pi

Use Raspberry Pi OS 64-bit (Bookworm or newer), ideally on a Pi 4 or 5.

```bash
sudo apt install espeak-ng python3-picamera2   # speech, and the camera module if you use one
python -m venv --system-site-packages .venv    # so the venv can see picamera2
source .venv/bin/activate
pip install -r requirements.txt
python -m scouter --camera picamera --headless
```

Face recognition is quick on a Pi. Object detection runs on a background thread, so the video and face greetings stay responsive even when objects take a second or two to update. The Pi camera path is written for picamera2 but hasn't been tested on real hardware yet.

## How it works

```
camera thread ──> newest frame ──> faces: YuNet detect → SFace embed → match against data/faces.npz
                                │                                            │
                                └─> object thread: YOLOX (640×640)            ├─> Announcer → Voice thread (pyttsx3 / espeak)
                                                   │                          │
                                                   └──────────────────────────┴─> HUD overlay
```

| File | |
|---|---|
| `scouter/__main__.py` | command line, main loop, enrolment |
| `scouter/faces.py` | face detection, embeddings, the face database |
| `scouter/objects.py` | YOLOX pre- and post-processing |
| `scouter/voice.py` | `Announcer` (what's worth saying) and `Voice` (speaking on a background thread) |
| `scouter/camera.py` | webcam / stream / Pi camera, always serving the newest frame |
| `scouter/hud.py` | the overlay |
| `scouter/models.py` | model download and checksum |

Two faces are the same person when the cosine similarity of their SFace embeddings is at least 0.363, the threshold the SFace authors give for their benchmark. Each person keeps all their enrolment samples, and a face is matched against the closest one.

## Tests

```bash
pip install pytest
python -m pytest
```

The tests cover the announcement rules (cooldowns, debouncing, phrasing), the face database, and YOLOX box decoding.

## Credits

Models from the [OpenCV model zoo](https://github.com/opencv/opencv_zoo): YuNet (MIT), SFace (Apache 2.0), YOLOX (Apache 2.0). They are downloaded at run time, not included in this repository.

## License

[MIT](LICENSE)
