# 🎨 AIrtify

This project demonstrates how to perform **Neural Style Transfer**, transforming an image by combining the content of one image with the artistic style of another using a pre-trained model from TensorFlow Hub.

---

## Output Example

| Original Content & Reference Style | Stylized Output |
|:--:|:--:|
| ![Content & Style](./Screenshot%202025-07-21%20012510.png) | ![Stylized Output](./Screenshot%202025-07-21%20012529.png) |

---

## 📌 About

The script:
- Loads a **content image** (e.g., cartoon ducks)
- Loads a **style reference image** (e.g., Monet-like painting)
- Uses TensorFlow Hub's `arbitrary-image-stylization-v1-256` model
- Produces a **stylized output image** using the reference style

---

## Model Used

- [Arbitrary Image Stylization V1](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2)
  - Developed by Google Magenta
  - Supports any content-style combination

---

### Install dependencies

```bash
pip install tensorflow tensorflow_hub matplotlib pillow

