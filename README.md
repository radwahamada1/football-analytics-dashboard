# ⚽ Football Analytics & Tactical Dashboard

An AI-powered **Football Analytics system** that uses computer vision to analyze football match videos, track players and the ball, identify teams, estimate possession, and generate tactical performance metrics.

Built with **YOLOv8, ByteTrack, Supervision, OpenCV, and Scikit-learn**.

---

## 🚀 Key Features

- 🏃 **Player Detection & Tracking** — Detects players using YOLOv8 and tracks them with ByteTrack.
- ⚽ **Ball Tracking & Interpolation** — Tracks the ball and estimates missing positions.
- 👕 **Team Classification** — Uses K-Means clustering on jersey colors to distinguish teams.
- 📊 **Possession Analysis** — Assigns the ball to the closest player and estimates team possession.
- 📐 **Perspective Transformation** — Maps player positions from the camera view to a top-down tactical pitch.
- ⚡ **Speed & Distance Estimation** — Estimates player movement speed and distance covered.
- 🖥️ **Tactical Dashboard** — Displays match statistics and analytics directly on the processed video.

---

## 🛠️ Tech Used

<p align="left">

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />

<img src="https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=yolo&logoColor=black" />

<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />

<img src="https://img.shields.io/badge/Supervision-6C5CE7?style=for-the-badge" />

<img src="https://img.shields.io/badge/ByteTrack-FF6F00?style=for-the-badge" />

<img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />

<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />

<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />

</p>

### 🔬 Computer Vision & Analytics

- **YOLOv8** (`ultralytics`) — Object detection
- **OpenCV** — Video processing and computer vision
- **Supervision** (`ByteTrack`) — Multi-object tracking
- **K-Means Clustering** (`scikit-learn`) — Team classification based on jersey colors
- **NumPy** — Numerical computations
- **Pandas** — Data processing and analysis
- **Homography Perspective Transformation** — Top-down tactical pitch mapping
- **Euclidean Distance & Speed Estimation** — Player movement analysis
- **Tactical Dashboard Overlay** — Real-time match analytics visualization
---

## 🧠 Pipeline

Football Match Video  
↓  
YOLOv8 Detection  
↓  
ByteTrack Tracking  
↓  
Player & Ball Tracking  
↓  
Team Color Classification  
↓  
Ball Assignment & Possession  
↓  
Perspective Transformation  
↓  
Speed & Distance Analysis  
↓  
Tactical Dashboard

---

## 📂 Project Structure

football_analytics/
│
├── data/
│   └── video/
│       └── sample.mp4
│
├── output_videos/
│
├── utils/
│   ├── ball_assigner.py
│   ├── ball_tracker.py
│   ├── color_assigner.py
│   ├── speed_and_distance_estimator.py
│   ├── tactical_dashboard.py
│   └── view_transformer.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md

---

## ▶️ Installation & Usage

### 1. Clone the repository

git clone repo

cd football-analytics

### 2. Install dependencies

pip install -r requirements.txt

### 3. Add your match video

Place your input video inside:

data/video/

### 4. Run the project

python main.py

The processed video will be saved in:

output_videos/

---

## 📊 Analytics

The system generates:

- 🏃 Player tracking
- ⚽ Ball tracking
- 👕 Team identification
- 📊 Ball possession
- ⚡ Player speed
- 📏 Distance covered
- 📐 Tactical positioning

---

## 🎯 Skills Demonstrated

**Computer Vision · Object Detection · Multi-Object Tracking · K-Means Clustering · Perspective Transformation · Motion Analysis · Sports Analytics**

---

## 💡 Project Goal

The goal of this project is to demonstrate how **AI and Computer Vision can transform raw football footage into structured tactical and performance insights.**

---

> ⚽ **Turning football match footage into meaningful tactical insights using AI and Computer Vision.**
