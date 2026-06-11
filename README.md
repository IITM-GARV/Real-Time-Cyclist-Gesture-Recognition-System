# RoadLink: Real-Time Cyclist Intent Recognition for Autonomous Vehicles

RoadLink is a real-time computer vision system designed to improve communication between cyclists and autonomous vehicles by detecting and classifying cyclist traffic signals from live video streams.

The system utilizes MediaPipe Holistic, OpenCV, and geometric pose analysis to recognize cyclist intentions through body posture, arm orientation, hand landmarks, and palm positioning. Temporal smoothing techniques are applied to improve stability and reduce false detections.

## Features

- Real-time cyclist gesture recognition
- Human pose estimation using MediaPipe Holistic
- Hand landmark tracking and palm orientation analysis
- Geometric gesture classification
- Temporal signal smoothing for stable predictions
- Live webcam-based operation
- Lightweight and privacy-preserving design

## Recognized Signals

- Left Turn
- Right Turn
- Stop
- Slow Down
- Cruising

## Technology Stack

- Python
- OpenCV
- MediaPipe Holistic
- NumPy

## Methodology

1. Capture live webcam video.
2. Extract body and hand landmarks using MediaPipe Holistic.
3. Compute geometric features such as:
   - Elbow angle
   - Forearm inclination
   - Arm orientation
   - Palm orientation
4. Apply rule-based gesture classification.
5. Use temporal smoothing buffers to reduce prediction noise.
6. Display recognized cyclist signals in real time.

## Applications

- Autonomous Vehicles
- Driver Assistance Systems
- Intelligent Transportation Systems
- Traffic Safety Monitoring
- Human Gesture Recognition

## Limitations

- Sensitive to poor lighting conditions
- Performance may decrease under motion blur
- Limited by the camera's field of view

## Future Improvements

- Night-time recognition using infrared cameras
- Multi-person gesture recognition
- Wider gesture vocabulary
- Vehicle-mounted deployment
- Integration with autonomous driving systems

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python roadlink.py
```

## Team

- ED23B002 – Adharsh SB
- ED23B021 – Shreyas
- ED23B022 – Garv

## License

MIT License

## Course Project

ED3010 – Human Factors in Design - IIT Madras
