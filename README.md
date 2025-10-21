#  VGG16-U-Net for DICOM Image-to-Image Translation (NAC → MAC)

This repository implements a VGG16-based U-Net for image-to-image translation on medical DICOM images, transforming Non-Attenuation-Corrected (NAC) PET images into Attenuation-Corrected (MAC) PET images. The workflow includes DICOM handling, normalization, data augmentation, model definition, training, and performance visualization.

---

##  Overview

The pipeline:
1. Loads and preprocesses DICOM images from two folders:
   - Unzipped_Mix/Mix/NAC (inputs)
   - Unzipped_Mix/Mix/MAC (targets)
2. Computes a global maximum pixel value for normalization across all slices.
3. Converts 2D grayscale slices into 3-channel RGB arrays.
4. Performs data augmentation using ImageDataGenerator.
5. Builds a VGG16-based U-Net with skip connections for image reconstruction.
6. Defines evaluation metrics — Dice, MAE, SSIM, and PSNR.
7. Trains and evaluates the model on the NAC→MAC translation task.
8. Visualizes predicted vs. ground-truth images and difference maps.

---

##  Directory Structure

Unzipped_Mix/
└── Mix/
    ├── NAC/   # Non-attenuation-corrected DICOM images (inputs)
    └── MAC/   # Attenuation-corrected DICOM images (targets)
Unzipped_Test_Mix/
└── Mix/
    ├── NAC/   # Test NAC images
    └── MAC/   # Test MAC images

---

## ⚙️ Requirements

Install dependencies (Python ≥ 3.8 recommended):

pip install tensorflow numpy pydicom scikit-image matplotlib opencv-python tqdm

Optional:
pip install keras

---

##  Model Architecture

- Encoder: Pretrained VGG16 (ImageNet weights, frozen) as feature extractor
- Decoder: Transposed convolutions + skip connections for upsampling
- Output: 3-channel image reconstructed via sigmoid activation

---

##  Data Preparation

The script reads all .dcm files from NAC and MAC directories:

files_nac = sorted([f for f in os.listdir(NAC_PATH) if f.endswith('.dcm')])
files_mac = sorted([f for f in os.listdir(MAC_PATH) if f.endswith('.dcm')])

- Calculates the global maximum intensity across both datasets.
- Normalizes each slice by dividing pixel values by this global_max.
- Expands grayscale images to 3 channels (RGB).

---

##  Data Augmentation

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='constant'
)
datagen.fit(X_train)

---

##  Evaluation Metrics

Metric | Description
-------|-------------
Dice Coefficient | Measures overlap between predicted and true masks
MAE (Mean Absolute Error) | 1 - mean(|y_true - y_pred|)
SSIM (Structural Similarity) | Measures perceptual similarity between images
PSNR (Peak Signal-to-Noise Ratio) | Quantifies reconstruction fidelity

---

##  Model Definition (VGG16 + U-Net Decoder)

The model uses VGG16 as the encoder and builds U-Net-like decoder layers with skip connections.

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, concatenate, BatchNormalization, Dropout

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
vgg16.trainable = False

Loss and optimizer:

model2.compile(optimizer=Adam(lr=0.001), loss='MSE', metrics=['accuracy'])

---

##  Training Visualization

After training, loss curves are plotted:

plt.plot(results2.history['loss'])
plt.plot(results2.history['val_loss'])
plt.legend(['loss','val_loss'])
plt.show()

---

##  Testing and Visualization

For each test DICOM image, the script:
- Predicts the MAC image from NAC input.
- Displays NAC, Predicted, Ground Truth, and Difference Map side by side.

---

##  Output

- X_train, Y_train arrays (augmented, normalized)
- predicted_images_12: Predicted outputs from the trained model
- Matplotlib visualizations of prediction results

---

##  Author

Aryan Golzaryan  
 aryan.golzaryan@gmail.com

---

