# 🎨 AIrtify

This project demonstrates how to perform **Neural Style Transfer**, transforming an image by combining the content of one image with the artistic style of another using a pre-trained model from TensorFlow Hub.

---

## Output Example

   
<img width="1194" height="609" alt="Screenshot 2025-07-21 012510" src="https://github.com/user-attachments/assets/9471cccc-6435-40be-a49a-13f5ed0e1839" />

    
<img width="638" height="425" alt="Screenshot 2025-07-21 012529" src="https://github.com/user-attachments/assets/4c5f1164-ca2c-4517-b295-570d77e809e6" />

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

