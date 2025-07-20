# VanGoghify
# 🎨 Van Gogh Style Transfer with TensorFlow

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project implements an end-to-end Neural Style Transfer (NST) pipeline that applies **Vincent van Gogh’s painting style** to any image using TensorFlow and VGG19. It includes fallbacks in case real Van Gogh images are unavailable by generating sample Van Gogh-style images from scratch.

---

## 🖼️ Example Output

| Original Image | Style Image | Stylized Result |
|----------------|-------------|------------------|
| ![Original](demo_images/original.png) | ![Style](demo_images/style1.png) | ![Stylized](demo_images/stylized.png) |

> *(Run the script to generate and save your own demo images)*

---

## 📌 Features

-  End-to-end **Neural Style Transfer** with fine-grained loss functions
-  Uses pretrained **VGG19** as the feature extractor
-  Automatically checks for real Van Gogh images or generates style samples
-  Supports **GPU acceleration** via TensorFlow
-  Clear separation between style and content loss
-  Fully **self-contained**, no external style libraries needed
