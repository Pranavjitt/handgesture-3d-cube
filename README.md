# handgesture-3d-cube

<img width="1918" height="1021" alt="image" src="https://github.com/user-attachments/assets/2fcafc4a-9c46-4896-972c-8c53f5d4f702" />

An interactive 3D cube application that responds to hand gestures captured through your webcam. Control and manipulate a rotating cube in real-time using natural hand movements.
3D Cube Hand Gesture Control 🎮✋
An interactive 3D cube visualization controlled by hand gestures using Python. This project combines computer vision and 3D graphics to create an intuitive gesture-based interface for manipulating a 3D cube in real-time.
🌟 Features

Real-time Hand Tracking: Detects and tracks hand movements using your webcam
Gesture Recognition: Recognizes various hand gestures to control the cube
3D Visualization: Smooth 3D cube rendering and rotation
Interactive Controls: Manipulate the cube's rotation, position, and scale using hand gestures
Visual Feedback: On-screen indicators showing detected hand landmarks and active gestures

🛠️ Technologies Used

Python 3.x
OpenCV: For video capture and image processing
MediaPipe: For hand detection and landmark tracking
Pygame/PyOpenGL: For 3D rendering and visualization
NumPy: For mathematical operations and transformations

📋 Prerequisites

Python 3.7 or higher
Webcam
Modern operating system (Windows/macOS/Linux)

🚀 Installation

Clone the repository:

bashgit clone https://github.com/yourusername/3d-cube-hand-gesture.git
cd 3d-cube-hand-gesture

Install required dependencies:

bashpip install opencv-python mediapipe pygame PyOpenGL numpy
💻 Usage
Run the main script:
bashpython main.py
```

- Allow camera access when prompted
- Position your hand in front of the camera
- Use different hand gestures to control the cube

## 🎯 Supported Gestures

*(Customize based on your implementation)*
- **Open Palm**: Rotate cube freely
- **Pinch**: Zoom in/out
- **Fist**: Stop rotation
- **Swipe Left/Right**: Rotate along Y-axis
- **Swipe Up/Down**: Rotate along X-axis

## 📁 Project Structure
```
3d-cube-hand-gesture/
│
├── main.py              # Main application script
├── hand_tracking.py     # Hand detection and gesture recognition module
├── cube_renderer.py     # 3D cube rendering logic
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
🤝 Contributing
Contributions are welcome! Feel free to:

Report bugs
Suggest new features
Submit pull requests

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

Built with assistance from Claude (Anthropic)
MediaPipe for hand tracking technology
OpenCV community for computer vision tools

📧 Contact
For questions or feedback, please open an issue on GitHub.
