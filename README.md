# 🔥 Forest Fire Detection System

> **AI-powered forest fire detection and monitoring platform using deep learning, real-time image classification, visual explainability, and geolocation-based monitoring.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-API-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![React](https://img.shields.io/badge/React-18%2B-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-Fast%20Build%20Tool-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind%20CSS-UI-06B6D4?logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)

---

## 📌 Overview

Forest fires can spread rapidly, causing severe damage to ecosystems, wildlife, infrastructure, and human communities.

This project provides an **AI-powered forest fire detection and monitoring system** that analyzes forest or satellite imagery using a deep-learning image classification model.

The system combines:

- 🧠 **MobileNetV2 transfer learning**
- 🔥 **Fire / No-Fire image classification**
- 📊 **Confidence scoring**
- 🧠 **Grad-CAM visual explainability**
- 📡 **Live camera monitoring simulation**
- 🗺️ **Camera geolocation visualization**
- 🚨 **Automated critical-risk alerts**
- 📜 **Detection history**
- 📈 **Detection statistics**
- ⚡ **React-based monitoring dashboard**
- 🔌 **Flask REST API**

The goal is to demonstrate how computer vision and modern web technologies can be combined into a practical early-warning system for forest-fire monitoring.

---

# ✨ Features

## 🧠 AI Fire Detection

The system uses **MobileNetV2 with transfer learning** to classify images into:

- 🔥 Fire
- 🌲 No Fire

The model processes images at **224 × 224 pixels** and uses MobileNetV2 preprocessing before inference.

---

## 🎯 Confidence-Based Risk Assessment

Every prediction returns:

- Detection result
- Confidence score
- Timestamp

The frontend uses the model's confidence to determine the corresponding risk level.

---

## 🔥 Grad-CAM Explainability

Instead of treating the model as a black box, the backend generates a **Grad-CAM heatmap** showing the regions of the image that contributed to the prediction.

This helps users understand:

> **"Where did the model see evidence of fire?"**

The explainability layer is particularly useful for demonstrating model transparency and building trust in AI-assisted detection.

---

## 📡 Live Monitoring

The dashboard includes a live-monitoring mode designed to simulate multiple forest monitoring cameras.

The system can:

- Monitor camera feeds
- Analyze incoming frames
- Update camera status
- Track the latest detection for each camera
- Trigger alerts for critical-risk detections

---

## 🗺️ Geolocation & Camera Mapping

Live camera detections are associated with predefined camera locations.

The dashboard visualizes the latest status of monitored locations on a map.

This allows operators to quickly identify:

- 🔥 Cameras detecting fire
- 🟢 Safe monitoring zones
- ⚠️ High-risk areas
- 📍 Camera locations

---

## 🚨 Automated Alerts

Critical-risk detections generate alerts in the dashboard.

Alerts contain:

- Location
- Risk level
- Confidence
- Detection timestamp

The system intentionally limits automatic alerts to **critical-risk events** to reduce unnecessary notifications.

---

## 📜 Detection History

The backend stores detection events in a lightweight JSON log.

The dashboard can display recent detections including:

- Filename
- Detection result
- Confidence
- Timestamp

The backend exposes the latest 20 history entries through the API.

---

## 📊 Model & System Statistics

The dashboard can display aggregate detection statistics including:

- Total images analyzed
- Fire detections
- Safe detections
- Average confidence

The training pipeline also generates evaluation metrics including:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────────┐
                         │       User / Operator    │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │     React Dashboard      │
                         │                          │
                         │ • Image Upload            │
                         │ • Live Monitoring          │
                         │ • Results                  │
                         │ • Alerts                   │
                         │ • History                  │
                         │ • Map                      │
                         └────────────┬─────────────┘
                                      │
                                  REST API
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │      Flask Backend       │
                         │                          │
                         │ /predict                 │
                         │ /health                  │
                         │ /history                 │
                         │ /stats                   │
                         │ /model-info              │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────┴─────────────┐
                         │                          │
                         ▼                          ▼
                ┌──────────────────┐      ┌─────────────────┐
                │   MobileNetV2    │      │    Grad-CAM     │
                │  Classification  │      │ Explainability  │
                └────────┬─────────┘      └─────────────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Fire / No Fire   │
                │ + Confidence     │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Alert / History  │
                │ / Statistics     │
                └──────────────────┘