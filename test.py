import os
from attacker_detector.is_human_speaker import is_human_speaker

path = 'audios/'
audios_path = [os.path.join(path, file) for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]

for audio_path in audios_path:
    print(audio_path, is_human_speaker(audio_path))