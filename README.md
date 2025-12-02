 # Arvyax Assignment: Hand Tracking Prototype (Classical CV)

## 📋 Overview
This is a real-time computer vision prototype that tracks a user's hand and detects proximity to a virtual object.
**Constraint Checklist:**
* ✅ **No MediaPipe / OpenPose:** Uses pure Color Segmentation & Math.
* ✅ **Real-time Performance:** Runs at >30 FPS on CPU.
* ✅ **Dynamic Logic:** Implements SAFE / WARNING / DANGER states based on Euclidean distance.

## 🎥 Demo Video
> *Watch the system in action below. Note the calibration sliders used to isolate the hand.*

https://github.com/user-attachments/assets/ee652686-5b38-4b28-bb66-e811e5dd8925

## 🛠️ Methodology (How it works)
Instead of using "Black Box" AI models, this solution builds a manual vision pipeline:

1.  **Preprocessing:** Converts the camera feed from BGR to **HSV Color Space**.
2.  **Segmentation:** Uses dynamic thresholding (controlled by sliders) to create a binary mask of the skin color.
3.  **Noise Removal:** Applies Morphological operations (Erosion/Dilation) to remove background noise (like walls or shadows).
4.  **Tracking:** Finds the largest contour (the hand) and calculates its **Centroid** $(x, y)$.
5.  **State Machine:**
    * Calculates Euclidean Distance between Hand Centroid and the Virtual Target.
    * Triggers UI overlays (Green/Yellow/Red) based on proximity thresholds.

## 🚀 How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Application:**
    ```bash
    python main.py
    ```

3.  **⚠️ IMPORTANT: Calibration Step**
    * When the app starts, two windows will appear.
    * Look at the **"Mask"** window (Black & White).
    * Adjust the **Hue/Sat/Val** sliders until your background is black and **only your hand is white**.
    * *Tip:* If the background is noisy, increase `Sat Min`.

## 📦 Requirements
* Python 3.x
* OpenCV (`opencv-python`)
* NumPy
