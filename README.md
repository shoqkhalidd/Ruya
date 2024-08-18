# Ruya

Ruya is an advanced voice assistant designed to assist visually impaired individuals by providing real-time object detection, distance estimation, and voice interaction capabilities.

## Features

- **Voice Activation**: Wake word detection using Picovoice Porcupine.
- **Speech Recognition**: Converts speech to text using OpenAI's Whisper model.
- **Text-to-Speech**: Converts text responses to speech using ElevenLabs API.
- **Object Detection and Tracking**: Detects and tracks objects in real-time using YOLO and DeepSort.
- **Distance Estimation**: Estimates the distance of detected objects using a pre-trained distance model.
- **Direction Guidance**: Provides directional guidance based on the position of detected objects.


## Installation

1. Get an API Key from Elevenlabs, Open-AI, and Porcupine.
2. Clone the repo
   ```sh
   git clone https://github.com/shoqkhalidd/Ruya.git
   ```
3. Install Python packages
   ```sh
   pip install -r requirements.txt
   ```
4. Add your API to OS Environment 
   ```sh
   export API_KEY = 'ENTER YOUR API'
   ```
5. Run the app
    ```sh
   python main.py track.py
   ```


## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contribution

This project was a collaborative effort by our team:

- [**Abdulaziz Alenazi**](https://linkedin.com/in/abdulaziz-alenazi)
- [**Badr Alanazi**](https://linkedin.com/in/badralanazix)
- [**Modar Alfadly** (Mentor)](https://linkedin.com/in/modar-alfadly)
