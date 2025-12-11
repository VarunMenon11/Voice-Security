# Voice Security (Research Prototype)

This project implements a lightweight speaker verification system built around GE2E-style embeddings, combined with a spoof detection module that identifies replay attacks, enabling access control only when the speaker sounds like the enrolled user and the speech was produced live.

---

## 1. Motivation & Background

Voice authentication is increasingly used in consumer devices, banking, and smart assistants, but most systems tend to focus only on matching the speaker’s identity. In practice, an attacker doesn’t even need to imitate the voice ‒ simply replaying a previous audio recording of the user can often bypass naive speaker-verification systems. This led me to look for methods that learn more discriminative speaker representations and also consider real-world spoofing scenarios.
During my exploration, I studied the paper “Generalized End-to-End Loss for Speaker Verification” by Wan et al. (Google AI Research, 2018), which proposes learning speaker embeddings directly from utterances. These embeddings make it possible to compare speakers in a continuous vector space using similarity metrics such as cosine distance. After reading the paper and implementing the GE2E approach, I became interested in testing how well such embedding-based verification would behave under simple real-world attacks, especially replayed samples recorded through another device.

This project is therefore a research prototype combining:
   o GE2E-style speaker verification
   o a simple replay-vs-live spoof classifier
   o and an access decision that accepts audio only if it sounds like the enrolled speaker and is predicted to be genuinely spoken, not replayed.

---

## 2. System Overview

(Explain the overall idea. How GE2E is used for embeddings, how the centroid works, how spoof detection works, and how the final decision is made.)

### Pipeline

(Insert a simple text diagram here)

---

## 3. Repository Structure

    ├── src/
    │   ├── ge2e_encoder.py         # GE2E-style model definition
    │   ├── utils_audio.py          # Audio loading and feature extraction helpers
    │   ├── feature_extraction.py   # Extract embeddings from LIVE/REPLAY audio
    │   ├── train_spoof.py          # Train LIVE vs REPLAY classifier
    │   ├── verify.py               # Main verification + spoof detection pipeline
    │   
    ├── config.yaml                 # Paths, thresholds, and parameters
    ├── requirements.txt            # Python dependencies
    └── README.md

## 4. Requirement & Installation

    # Clone the repo
    git clone https://github.com/VarunMenon11/Voice-Security.git
    cd Voice-Security
    
    # (Optional) create virtual env
    python -m venv .Voice_Security_Enviroment
    .\.venv\Scripts\activate   # Windows
    # source .venv/bin/activate   # Linux/macOS
    
    # Install dependencies
    pip install -r requirements.txt

## 5. Data Setup


    Real Voice Dataset/            # LIVE recordings (User)
        User_01.wav
        User_02.wav
        ...
    Recorded Voice Dataset/        # REPLAY recordings (User playback)
        User_Recorded_01.wav
        User_Recorded_02.wav
        ...
    
    (optional)
    data/eval/live/                # extra Varun test samples
    data/eval/other/               # friends / impostor voices


## 6. Configuration(Congif.yaml)

## 7. How to Run?

## 8. Current Result

## 9.Future Work

