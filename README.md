
# Lightweight Onboard AI for Mars Rovers

### A TinyML Approach to Environmental Prediction

**Author:** Neharika Kotamaraju
**Institution:** Amrita Vishwa Vidyapeetham, Coimbatore
**Department:** Electrical and Electronics Engineering

---

## Project Overview

Mars exploration demands autonomous systems capable of making real-time decisions under harsh environmental conditions and communication delays.

This project presents a **TinyML-based lightweight AI system** for predicting Martian environmental parameters onboard a rover.

The models are optimized for edge deployment using:

* TensorFlow Lite
* Edge Impulse
* Quantization techniques
* Embedded system compatibility testing

The system is designed to operate on microcontrollers and edge AI hardware platforms.

---

## Dataset

The dataset is derived from the **Mars Rover Environmental Monitoring Station (REMS)** onboard NASA's Curiosity rover.

### Features Used:

* max_ground_temp (°C)
* min_ground_temp (°C)
* max_air_temp (°C)
* min_air_temp (°C)
* mean_pressure (Pa)
* UV_Level_high
* UV_Level_low
* UV_Level_moderate
* UV_Level_very_high
  To predict max ground Temp

Cleaned dataset size: 3170 complete records

---

## Models Implemented

### 1. Deep Neural Network (DNN)

* Dense architecture
* ReLU activation
* Linear output layer
* Optimized using Adam
* Loss: Mean Squared Error (MSE)

### Architecture:

* Input Layer: 128 neurons (ReLU)
* Hidden Layer 1: 64 neurons (ReLU)
* Hidden Layer 2: 32 neurons (ReLU)
* Output Layer: 1 neuron (Linear)

### Performance:

* Float32 Accuracy: 93.57%
* INT8 Quantized Accuracy: 89.88%
* Latency: ~1 ms
* RAM Usage: ~1.2 KB
* Flash Size: ~11.1 KB

---

### 2. 1D Convolutional Neural Network (1D CNN)

* 1D Convolution layer for sequential feature extraction
* ReLU activation
* Dense output layer
* Regression-based prediction

### Performance:

* Float32 Accuracy: 82.10%
* INT8 Quantized Accuracy: 76.13%
* Latency: 1–10 ms
* RAM Usage: ~1.2 KB
* Flash Size: ~11.1 KB

---

## Model Development Workflow

### Step 1: Data Preprocessing (Google Colab)

* Load dataset using Pandas
* Handle missing values
* Drop incomplete rows
* Perform summary statistics
* Standardize features using StandardScaler

### Step 2: Model Training

* Train DNN and 1D CNN models
* 100 epochs
* Batch size: 32
* Optimizer: Adam
* Loss: MSE
* Evaluation metrics:

  * MSE
  * MAE
  * R² Score

### Step 3: Model Conversion

Convert trained models to TensorFlow Lite format:

```
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

Saved as:

* dnn_model.tflite
* cnn_model.tflite

---

## Edge Impulse Integration

After training in Google Colab:

1. Upload dataset to Edge Impulse Studio
2. Upload trained TFLite model
3. Configure impulse design
4. Enable:

   * Data Explorer
   * PCA dimensionality reduction
   * GPU training (if available)
5. Perform quantization (INT8)
6. Test deployment metrics
7. Generate deployment binaries

### Edge Impulse Public Project:
https://studio.edgeimpulse.com/public/675626/live  for DNN Model

https://studio.edgeimpulse.com/public/675947/live  for 1D CNN Regression




---

## TinyML Optimizations Used

* Post-training quantization (INT8)
* TensorFlow Lite conversion
* Reduced memory footprint
* Low latency inference
* Edge profiling for RAM & Flash usage

---

## Deployment Configurations Supported

The trained models can be deployed to:

### Microcontrollers

* Arduino Nano 33 BLE Sense
* STM32 (CubeMX CMSIS Pack)
* Renesas RA6M5
* Raspberry Pi Pico (RP2040 / RP2350)
* Nordic nRF9161 DK
* Ambiq Apollo 5 EVB

### Edge AI Boards

* NVIDIA Jetson Nano
* NVIDIA Jetson Xavier
* NVIDIA Jetson Orin
* TI AM62A
* TI AM68A
* TI TDA4VM
* BrainChip Akida (AKD1000 / AKD1500)
* ARM Ethos-U55 / U65 / U85 NPUs
* Qualcomm Dragonwing IQ 9075

### Deployment Formats

* C++ Library
* Arduino Library
* CMSIS Pack
* Zephyr Module
* WebAssembly
* TensorRT Library
* Docker Container
* Linux (AARCH64 / x86)
* Android C++ Library

---

## Edge Deployment Pipeline

Sensor Data → Preprocessing → DNN / 1D CNN →
Quantization → TensorFlow Lite →
Edge Impulse Profiling →
Embedded Deployment →
Real-time Rover Decision Control

---

## Key Advantages

* Real-time inference (<1 ms)
* Extremely low RAM footprint
* Microcontroller-compatible
* Suitable for deep-space missions
* Energy-efficient
* Autonomous decision support

---

## Applications

* Mars rover onboard intelligence
* Environmental monitoring
* Space robotics
* Low-power embedded AI systems
* Autonomous planetary exploration

---

## Future Improvements

* Pruning-based model compression
* Radiation-hardened MCU testing
* Multi-parameter environmental prediction
* On-device continual learning

---

## License

This project is released under BSD 3-Clause License (as per Edge Impulse project settings).

