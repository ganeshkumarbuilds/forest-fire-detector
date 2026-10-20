# 🔥 Forest Fire Detection System

An AI-powered forest fire detection and monitoring platform combining deep learning, explainable AI, geolocation visualization, and a React-based monitoring dashboard.

The system uses **MobileNetV2 transfer learning** to classify images as fire or no-fire, provides prediction confidence, generates Grad-CAM visualizations, and presents detection information through a Flask REST API and React frontend.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Flask-REST%20API-000000?logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/React-18%2B-61DAFB?logo=react&logoColor=black" alt="React">
  <img src="https://img.shields.io/badge/Vite-Build%20Tool-646CFF?logo=vite&logoColor=white" alt="Vite">
  <img src="https://img.shields.io/badge/Tailwind%20CSS-UI-06B6D4?logo=tailwindcss&logoColor=white" alt="Tailwind CSS">
</p>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [API Overview](#-api-overview)
- [Getting Started](#-getting-started)
- [Model Evaluation](#-model-evaluation)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📌 Overview

Forest fires can cause significant damage to ecosystems, wildlife, infrastructure, and nearby communities. Early identification of potential fire incidents can help monitoring teams investigate suspicious conditions more quickly.

The **Forest Fire Detection System** is a computer vision project designed to explore how deep learning and modern web technologies can work together in a forest-monitoring application.

It combines an image classification model with a web-based monitoring interface, allowing users to analyze images, inspect predictions, review detection history, and visualize monitoring locations.

### 🎯 Project Objectives

- Apply transfer learning to a binary fire-image classification problem.
- Provide prediction confidence alongside classification results.
- Improve model interpretability through Grad-CAM visualizations.
- Build a REST API for model inference and detection information.
- Develop a React dashboard for image analysis and monitoring.
- Visualize predefined camera locations and their latest detection status.
- Present detection history, aggregate statistics, and critical-risk alerts.

---

## 🚀 Live Demo

**Frontend:** [Open the Forest Fire Detection Dashboard](https://frontend-fire-detector.vercel.app/)

> **Note:** Replace the example URL above with the actual deployed frontend URL from this repository's GitHub About section before publishing. Confirm that the deployment is accessible and that its backend-dependent features work.

---

## ✨ Key Features

### 🧠 1. AI-Powered Fire Detection

The system uses **MobileNetV2 with transfer learning** to classify images into two categories:

- 🔥 Fire
- 🌲 No Fire

The documented inference pipeline:

1. Accepts an input image.
2. Resizes the image to `224 × 224` pixels.
3. Applies MobileNetV2-compatible preprocessing.
4. Runs the image through the trained model.
5. Produces a classification result and confidence score.

The model is intended for image-based fire detection experiments and demonstrations.

### 🎯 2. Confidence-Based Risk Assessment

The prediction workflow exposes information associated with each detection:

- Classification result
- Prediction confidence
- Detection timestamp

The frontend uses prediction confidence to determine the displayed risk level.

**Important:** Model confidence is not the same as the real-world probability of a fire. Risk thresholds should be evaluated against validation data before the system is used for consequential decisions.

### 🔥 3. Explainable AI with Grad-CAM

Deep-learning predictions can be difficult to interpret. This project incorporates **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualize image regions that influence a model prediction.

The visualization helps investigate questions such as:

- Which image regions influenced the classification?
- Is the model focusing on the apparent fire region?
- Could the prediction be influenced by irrelevant background features?

Grad-CAM is a diagnostic aid, not proof that a prediction is correct.

### 📡 4. Live Camera Monitoring Simulation

The dashboard includes a live-monitoring mode designed to simulate multiple forest-monitoring cameras.

Documented monitoring capabilities include:

- Displaying camera information
- Analyzing incoming frames
- Updating camera status
- Tracking the latest detection for each camera
- Presenting critical-risk alerts

The monitoring simulation should not be confused with a fully deployed physical camera network.

### 🗺️ 5. Geolocation and Camera Mapping

The system associates monitoring detections with predefined camera locations and visualizes their status on a map.

The interface is designed to help users inspect:

- Camera locations
- Locations associated with fire detections
- Safe-status monitoring locations
- Locations flagged as high risk

The accuracy of displayed locations depends on the configured camera-location data.

### 🚨 6. Critical-Risk Alerts

The dashboard can display alerts for detections that meet the configured critical-risk criteria.

Alert information includes:

- Location
- Risk level
- Prediction confidence
- Detection timestamp

The system limits automatic alerts to configured critical-risk events to reduce unnecessary notifications.

### 📜 7. Detection History

The backend records detection events in a lightweight JSON log.

The history interface can display:

- Image filename
- Classification result
- Confidence score
- Timestamp

The API exposes the latest 20 history entries.

### 📊 8. Model and System Statistics

The dashboard presents aggregate detection information, including:

- Total images analyzed
- Number of fire detections
- Number of safe detections
- Average prediction confidence

The training pipeline also generates evaluation outputs for:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix

Actual evaluation results should be reported only after running the evaluation pipeline.

---

## 🏗️ System Architecture

The project is organized into three primary components:

1. **React frontend:** User interface, image submission, monitoring, history, alerts, and map visualization.
2. **Flask backend:** REST API, prediction handling, model integration, Grad-CAM generation, and detection information.
3. **ML model:** MobileNetV2-based image classification.

```text
                    ┌─────────────────────────┐
                    │      User / Operator    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     React Frontend      │
                    │                         │
                    │  Image Upload           │
                    │  Monitoring Dashboard   │
                    │  Detection Results      │
                    │  Alerts and History     │
                    │  Geolocation Map        │
                    └────────────┬────────────┘
                                 │
                            HTTP / REST
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     Flask Backend       │
                    │                         │
                    │  Image Preprocessing    │
                    │  Model Inference        │
                    │  Grad-CAM Generation    │
                    │  Detection History      │
                    │  Statistics and Alerts  │
                    └────────────┬────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
                  ▼                             ▼
       ┌────────────────────┐       ┌────────────────────┐
       │     MobileNetV2    │       │      Grad-CAM      │
       │                    │       │                    │
       │  Image             │       │  Prediction        │
       │  Classification    │       │  Visualization     │
       └──────────┬─────────┘       └────────────────────┘
                  │
                  ▼
       ┌────────────────────┐
       │  Fire / No Fire    │
       │  Confidence Score  │
       └──────────┬─────────┘
                  │
                  ▼
       ┌────────────────────┐
       │  Detection Events  │
       │  History / Alerts  │
       │  Statistics        │
       └────────────────────┘
```

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Programming language | Python | Model integration and backend development |
| Deep learning | TensorFlow / Keras | Model training and inference |
| Model architecture | MobileNetV2 | Transfer-learning-based image classification |
| Explainable AI | Grad-CAM | Visualization of image regions influencing predictions |
| Backend | Flask | REST API and inference services |
| Frontend | React | Interactive monitoring interface |
| Frontend tooling | Vite | Development server and build tooling |
| Styling | Tailwind CSS | User interface styling |
| Data logging | JSON | Lightweight detection-history storage |
| Deployment | Vercel and Render configuration | Frontend and backend deployment |

---

## 📂 Project Structure

The repository separates the backend, frontend, and machine-learning components.

```text
forest-fire-detector/
│
├── backend/
│   └── Flask application, API endpoints,
│       inference integration, and detection handling
│
├── frontend/
│   └── React application, dashboard,
│       monitoring interface, and visualization
│
├── ml-model/
│   └── Model training, evaluation,
│       and machine-learning resources
│
├── .gitignore
├── .python-version
├── README.md
└── render.yaml
```

> The tree above summarizes the top-level organization. Consult the actual directories for the complete file structure.

---

## ⚙️ How It Works

### Image Classification Workflow

1. A user submits an image through the frontend.
2. The frontend sends the image to the Flask backend.
3. The backend preprocesses the image for the model.
4. MobileNetV2 performs image classification.
5. The backend prepares the prediction and confidence information.
6. Grad-CAM generates an explanatory visualization when requested or supported by the inference workflow.
7. The detection information is returned to the frontend.
8. The dashboard displays the result and updates the relevant monitoring information.

### Monitoring Workflow

1. The dashboard presents configured monitoring cameras.
2. The monitoring simulation processes incoming frames according to its implementation.
3. Camera status is updated using detection information.
4. Detection events can be added to the history log.
5. Critical-risk detections can generate dashboard alerts.
6. The interface displays detection statistics and camera locations.

---

## 🔌 API Overview

The Flask backend exposes endpoints for prediction, service health, history, statistics, and model information.

| Endpoint | Purpose |
|---|---|
| `/predict` | Submit an image for fire classification |
| `/health` | Check backend service health |
| `/history` | Retrieve recent detection history |
| `/stats` | Retrieve aggregate detection statistics |
| `/model-info` | Retrieve available model information |

The exact HTTP methods, request fields, response schemas, and error codes depend on the implementation.

Refer to the backend source code for the authoritative API contract.

---

## 🚀 Getting Started

You can run the application locally after installing its dependencies and configuring the model resources.

### Prerequisites

- Python compatible with the backend dependencies
- Node.js and npm
- Git
- The model files required by the inference pipeline

The repository currently includes a `.python-version` file and a `render.yaml` deployment configuration. Use the actual project configuration to determine the supported runtime and dependency versions.

### 1. Clone the Repository

```bash
git clone https://github.com/ganeshkumarbuilds/forest-fire-detector.git
cd forest-fire-detector
```

### 2. Set Up the Backend

Open a terminal in the project root.

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies using the backend's dependency manifest.

For example, if `backend/requirements.txt` exists:

```bash
pip install -r requirements.txt
```

> If the backend uses a different dependency manifest, follow that manifest instead. Do not create a new requirements file containing guessed dependencies.

Start the Flask application using its actual entry point and configuration.

For example, if the application entry point is `app.py` and it supports Flask's development server:

```bash
flask --app app run --debug
```

The command above is an example. Adjust it to match the backend entry point.

### 3. Set Up the Frontend

Open a second terminal:

```bash
cd frontend
```

Install the Node.js dependencies:

```bash
npm install
```

Start the Vite development server:

```bash
npm run dev
```

Open the local address printed by Vite in your terminal.

### 4. Verify the Integration

After starting the frontend and backend:

1. Confirm that the frontend loads.
2. Confirm that the backend health endpoint responds.
3. Submit a test image.
4. Verify that the backend returns a classification result.
5. Check whether confidence information is displayed correctly.
6. Test Grad-CAM visualization if available.
7. Verify that history and statistics update as expected.

**Important:** The example commands may require adjustment to match the actual entry points, dependency files, model paths, and configuration in your repository.

---

## 📈 Model Evaluation

The project includes an evaluation workflow that generates standard classification metrics.

### Metrics

| Metric | What It Measures |
|---|---|
| Accuracy | Proportion of correctly classified examples |
| Precision | Proportion of predicted fire cases that are actually fire cases |
| Recall | Proportion of actual fire cases correctly detected |
| F1 score | Harmonic mean of precision and recall |
| Confusion matrix | Distribution of correct and incorrect classifications |

For a fire-detection application, recall is particularly important because false negatives can represent missed fire images. Precision is also important because false positives can generate unnecessary alerts.

A meaningful evaluation should use a held-out test set that was not used for training or model selection.

Report the dataset, class distribution, test-set size, evaluation procedure, and actual measured results when available.

**No accuracy or performance figures are claimed here because verified evaluation results have not been provided.**

---

## ⚠️ Limitations

This project is an AI-based monitoring prototype, not a certified fire-detection or emergency-response system.

Important limitations include:

- Image classification does not establish whether a real-world fire is present at a particular location.
- Prediction confidence is not necessarily a calibrated probability of fire.
- Model performance can vary with lighting, smoke, weather, image quality, camera angle, and environmental conditions.
- Grad-CAM visualizations do not guarantee that a prediction is correct or explain the model completely.
- The monitoring simulation does not, by itself, establish integration with physical camera infrastructure.
- Map accuracy depends on the quality and correctness of the configured location data.
- JSON-based detection logging may not be suitable for high-volume production deployments.
- Reliable real-world deployment would require broader testing, operational monitoring, and an appropriate human-verification process.

The system should not be used as the sole basis for emergency decisions.

---

## 🔮 Future Improvements

Potential improvements include:

- [ ] Document measured test-set performance and class-wise evaluation results.
- [ ] Add additional examples of correct and incorrect predictions.
- [ ] Evaluate robustness under different lighting and environmental conditions.
- [ ] Calibrate prediction confidence and validate risk thresholds.
- [ ] Add automated backend tests for prediction and history endpoints.
- [ ] Add frontend tests for important dashboard workflows.
- [ ] Improve monitoring reliability and error handling.
- [ ] Introduce persistent database storage for larger detection histories.
- [ ] Add structured logging and deployment health monitoring.
- [ ] Evaluate performance on representative real-world camera imagery.
- [ ] Document API request and response examples.
- [ ] Add screenshots of the dashboard, detection results, and Grad-CAM visualizations.

---

## 🤝 Contributing

Contributions, suggestions, and constructive feedback are welcome.

To contribute:

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Run the relevant tests.
5. Submit a pull request describing the changes.

For substantial changes, open an issue first to discuss the proposed improvement.

---

## 📄 License

Check the repository for a `LICENSE` file before reusing or redistributing this project.

If no license has been added, the repository should not be assumed to permit unrestricted reuse.

---

## 👨‍💻 Author

**Ganesh Kumar**

- GitHub: [@ganeshkumarbuilds](https://github.com/ganeshkumarbuilds)
- Project Repository: [Forest Fire Detection System](https://github.com/ganeshkumarbuilds/forest-fire-detector)

---

<p align="center">
  <b>Building practical AI applications through computer vision, explainable AI, and full-stack development.</b>
</p>