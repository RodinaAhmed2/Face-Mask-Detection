# 😷 **Real-Time Face Mask Detection using CNN & AI**  

---

## 🧾 **Project Overview**

The **Face Mask Detection** project is an **AI-based computer vision system** designed to automatically detect whether a person is wearing a **face mask** or not in **real time**.  

Using a **Convolutional Neural Network (CNN)**, the model analyzes facial images and classifies them into two categories:
- 🟢 **With Mask**
- 🔴 **Without Mask**

Developed to support **public safety and health monitoring** during and beyond the COVID-19 pandemic,  
this project can be integrated into **surveillance cameras**, **access control systems**, and **public places** to ensure compliance with mask-wearing policies.

The system achieves **99.01% training accuracy** and **93.38% validation accuracy**, demonstrating **high reliability and efficiency**.  
It is built using **TensorFlow**, **Keras**, and **OpenCV**, and includes a **user-friendly GUI** developed with **Tkinter**.

---

## 🎯 **Main Objectives**
✅ Detect mask usage in real-time video or images using **deep learning**  
⚙️ Build a **lightweight CNN model** that balances accuracy and speed  
🪟 Develop an **interactive GUI** for users to upload images and see detection results  
📹 Support **integration with cameras** or live video feeds  

---

## 📸 **Demo & GUI**  
This project includes a simple **Graphical User Interface (GUI)** built with **Tkinter** for real-time predictions on images.  

*(You can insert screenshots or GIFs below to visualize your interface)*  

![GUI Demo Image](#)  
![Prediction Example](#)

---

## ⚙️ **Technologies Used**  
| Technology | Purpose |
|-------------|----------|
| 🐍 **Python 3.x** | Main programming language |
| 🧠 **TensorFlow & Keras** | Build and train the CNN model |
| 👁️ **OpenCV** | Image preprocessing and video capture |
| 🪟 **Tkinter** | GUI development |
| 📊 **Matplotlib & Seaborn** | Data visualization |
| ☁️ **Google Colab** | GPU-accelerated training |

---

## 🧠 **Model Architecture**

A custom **lightweight CNN** was built to ensure **fast inference** and **high accuracy** suitable for real-time use.  

| Layer | Description |
|:------|:-------------|
| **Input** | 100×100 RGB Image |
| **Conv2D (32, 3×3, ReLU)** → **MaxPooling2D (2×2)** | Extracts low-level features |
| **Conv2D (64, 3×3, ReLU)** → **MaxPooling2D (2×2)** | Extracts high-level features |
| **Flatten** | Converts 2D maps into 1D |
| **Dense (128, ReLU)** → **Dense (2, Softmax)** | Classifies “Mask” vs “No Mask” |

**Training Configuration**
- 🔹 Optimizer: **Adam**  
- 🔹 Learning Rate: **0.001**  
- 🔹 Loss Function: **Categorical Crossentropy**  
- 🔹 Epochs: **10**  
- 🔹 Batch Size: **32**

---

## 📈 **Performance & Results**

| Metric | Accuracy |
|:--------:|:---------:|
| 🧩 **Training Accuracy** | **99.01%** |
| 🧪 **Validation Accuracy** | **93.38%** |

### 🔹 Confusion Matrix (Validation)
|                | Predicted: With Mask | Predicted: Without Mask |
|:---------------|:--------------------:|:------------------------:|
| **Actual: With Mask** | ✅ 709 (TP) | ❌ 41 (FN) |
| **Actual: Without Mask** | ❌ 59 (FP) | ✅ 702 (TN) |

*(Insert your confusion matrix heatmap here)*  
![Uploading with_mask_3340.jpg…]()

---

## 📊 **Dataset**

The dataset is divided into two main categories:  
- 🟢 `with_mask/` → Images of people wearing masks  
- 🔴 `without_mask/` → Images of people without masks  

**Split:** 80% Training | 20% Validation  
**Preprocessing:** All images resized to **100×100 pixels** and normalized.  

🔗 Recommended Dataset: [Face Mask Detection Dataset on Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

---

## 🚀 **Limitations & Future Work**

### 🔸 Current Limitations
- Limited diversity in **lighting**, **ethnicity**, and **background conditions**.  
- Slightly struggles with **partially occluded faces** or **non-standard masks**.

### 🔹 Future Enhancements
- **Data Augmentation:** Apply rotation, zoom, and brightness changes.  
- **Transfer Learning:** Use pre-trained models like *MobileNetV2*.  
- **Real-Time Video Detection:** Extend GUI to live webcam detection.  

---

## 📄 **License**
📜 This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for full details.

---

## 🧑‍💻 Author 
- *Rodina Ahmed* → [GitHub Profile](https://github.com/RodinaAhmed) 
- *Farida Ayman* → [GitHub Profile](https://github.com/FaridaAyman) 
- *Nada Attia* → [GitHub Profile](https://github.com/NadaAttia04)
---

⭐ *If you like this project, don't forget to star the repository and support our work!* 🌟
