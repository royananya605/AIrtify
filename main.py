# Van Gogh Style Transfer Model - Fixed Dtype Issues
# This version fixes the float16/float32 dtype mismatch error

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import requests
from io import BytesIO
import random
from collections import Counter

print(f"TensorFlow version: {tf.__version__}")
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Ensure consistent float32 dtype throughout
tf.keras.backend.set_floatx('float32')

# Step 1: Load Van Gogh Dataset CSV
print("=== LOADING VAN GOGH DATASET CSV ===")
csv_path = "VanGoghPaintings.csv"

try:
    van_gogh_df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(van_gogh_df)} Van Gogh paintings metadata")
    print("Columns:", list(van_gogh_df.columns))
    print("\nClass distribution:")
    print(van_gogh_df['class_name'].value_counts())
    print("\nFirst few paths:")
    print(van_gogh_df['image_path'].head())
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    exit()

# Step 2: Check for actual image files
print("\n=== CHECKING IMAGE AVAILABILITY ===")

def check_local_images(df, possible_paths):
    """Check multiple possible local paths for images"""
    found_images = []

    for base_path in possible_paths:
        print(f"Checking path: {base_path}")
        if not os.path.exists(base_path):
            print(f"  ❌ Directory not found: {base_path}")
            continue

        count = 0
        for idx, row in df.head(100).iterrows():
            filename = os.path.basename(row['image_path'])
            possible_file = os.path.join(base_path, row['class_name'], filename)

            if os.path.exists(possible_file):
                found_images.append((idx, possible_file))
                count += 1

        print(f"  Found {count} images in first 100 entries")
        if count > 0:
            break

    return found_images

# Possible local paths where images might be
possible_paths = [
    "./van-gogh-paintings",
    "./VanGoghPaintings",
    "./images",
    "./dataset",
    "."
]

found_images = check_local_images(van_gogh_df, possible_paths)
print(f"✓ Found {len(found_images)} local images")

# Step 3: Alternative - Use sample Van Gogh style images
def create_sample_van_gogh_images():
    """Create sample Van Gogh style images for demonstration"""
    print("\n=== CREATING SAMPLE VAN GOGH STYLE IMAGES ===")
    print("Since local images aren't available, creating Van Gogh-inspired samples...")

    sample_images = []

    for i in range(5):
        # Create more realistic Van Gogh-style base
        img = np.zeros((256, 256, 3), dtype=np.float32)  # Explicitly set dtype

        # Create swirling patterns more characteristic of Van Gogh
        y, x = np.mgrid[0:256, 0:256]

        if i == 0:  # Starry Night inspired - swirling sky
            angle = np.arctan2(y - 128, x - 128)
            radius = np.sqrt((x - 128)**2 + (y - 128)**2)
            swirl = np.sin(angle * 3 + radius * 0.05) * 0.5 + 0.5

            img[:, :, 0] = 0.1 + 0.2 * swirl
            img[:, :, 1] = 0.2 + 0.4 * swirl
            img[:, :, 2] = 0.4 + 0.4 * (1 - swirl)

            star_positions = [(50, 80), (200, 60), (180, 120), (100, 40)]
            for sy, sx in star_positions:
                img[sy-3:sy+3, sx-3:sx+3, 1] = 0.9

        elif i == 1:  # Sunflowers inspired
            center_y, center_x = 128, 128
            angle = np.arctan2(y - center_y, x - center_x)
            radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)

            petals = np.sin(angle * 8) * 0.5 + 0.5
            center_mask = radius < 60

            img[:, :, 0] = 0.8 + 0.2 * petals
            img[:, :, 1] = 0.9
            img[:, :, 2] = 0.1 + 0.2 * petals

            img[center_mask, 0] = 0.3
            img[center_mask, 1] = 0.2
            img[center_mask, 2] = 0.1

        elif i == 2:  # Café terrace inspired
            for line_y in range(0, 256, 20):
                slope = (line_y - 128) / 256
                for px in range(256):
                    py = int(128 + slope * (px - 128))
                    if 0 <= py < 256:
                        img[py-2:py+2, px, :] = [0.6, 0.4, 0.1]

            img[:128, :, :] = [0.1, 0.1, 0.4]
            img[128:, :, 0] = 0.4 + 0.3 * np.random.rand(128, 256).astype(np.float32)
            img[128:, :, 1] = 0.3 + 0.2 * np.random.rand(128, 256).astype(np.float32)
            img[128:, :, 2] = 0.1

        else:  # Wheat fields / countryside
            for row in range(0, 256, 4):
                wave = np.sin(np.arange(256) * 0.1) * 10
                for col in range(256):
                    wave_row = int(row + wave[col])
                    if 0 <= wave_row < 256:
                        img[wave_row-1:wave_row+2, col, :] = [0.6, 0.7, 0.2]

            img[:100, :, :] = [0.4, 0.6, 0.8]

            tree_positions = [50, 150, 200]
            for tree_x in tree_positions:
                img[80:200, tree_x-2:tree_x+2, :] = [0.2, 0.4, 0.1]

        # Add texture
        texture = np.random.normal(0, 0.03, (256, 256, 3)).astype(np.float32)
        img = np.clip(img + texture, 0, 1).astype(np.float32)

        brush_texture = np.random.rand(256, 256, 3).astype(np.float32) * 0.1
        img = np.clip(img + brush_texture, 0, 1).astype(np.float32)

        sample_images.append(img)

    return np.array(sample_images, dtype=np.float32)

# Use found images or create samples
if len(found_images) > 0:
    print(f"\n=== LOADING {len(found_images)} LOCAL VAN GOGH IMAGES ===")
    van_gogh_images = []
    image_info = []

    for idx, image_path in found_images[:20]:
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((256, 256), Image.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            van_gogh_images.append(img_array)
            image_info.append({
                'path': image_path,
                'class': van_gogh_df.iloc[idx]['class_name'],
                'source': 'local'
            })
        except Exception as e:
            print(f"Error loading {image_path}: {e}")

    van_gogh_images = np.array(van_gogh_images, dtype=np.float32)
    print(f"✓ Loaded {len(van_gogh_images)} local Van Gogh images")

else:
    print("\n=== USING SAMPLE VAN GOGH STYLE IMAGES ===")
    van_gogh_images = create_sample_van_gogh_images()
    image_info = [
        {'class': 'Starry Night Style', 'source': 'generated'},
        {'class': 'Sunflower Style', 'source': 'generated'},
        {'class': 'Café Terrace Style', 'source': 'generated'},
        {'class': 'Mixed Style 1', 'source': 'generated'},
        {'class': 'Mixed Style 2', 'source': 'generated'}
    ]
    print(f"✓ Created {len(van_gogh_images)} sample Van Gogh style images")

print(f"Dataset shape: {van_gogh_images.shape}")

# Step 4: Display Van Gogh style images
print("\n=== VAN GOGH STYLE SAMPLES ===")
if len(van_gogh_images) > 0:
    fig, axes = plt.subplots(1, min(5, len(van_gogh_images)), figsize=(15, 3))
    if len(van_gogh_images) == 1:
        axes = [axes]

    for i in range(min(5, len(van_gogh_images))):
        if len(van_gogh_images) > 1:
            ax = axes[i]
        else:
            ax = axes[0]
        ax.imshow(van_gogh_images[i])
        ax.set_title(f"{image_info[i]['class']}", fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Step 5: Van Gogh Style Transfer Model - FIXED VERSION
class VanGoghStyleTransfer:
    def __init__(self, content_layers, style_layers):
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_style_layers = len(style_layers)
        self.num_content_layers = len(content_layers)

        self.vgg = self.vgg_layers(style_layers + content_layers)
        self.vgg.trainable = False

    def vgg_layers(self, layer_names):
        """Creates a vgg model that returns intermediate output values."""
        vgg = VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in layer_names]
        model = tf.keras.Model([vgg.input], outputs)
        return model

    def gram_matrix(self, input_tensor):
        """Calculate Gram matrix for style loss"""
        # Ensure consistent dtype - FIXED
        input_tensor = tf.cast(input_tensor, tf.float32)
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result/(num_locations)

    def style_content_loss(self, outputs, style_targets, content_targets,
                          style_weight=1e-2, content_weight=1e4):
        """Calculate the total loss - FIXED VERSION"""
        style_outputs = outputs[:self.num_style_layers]
        content_outputs = outputs[self.num_style_layers:]

        # Calculate style loss using Gram matrices
        style_losses = []
        for i in range(len(style_outputs)):
            style_out = tf.cast(style_outputs[i], tf.float32)
            style_gram = self.gram_matrix(style_out)
            style_target = tf.cast(style_targets[i], tf.float32)
            loss = tf.reduce_mean((style_gram - style_target)**2)
            style_losses.append(loss)

        style_loss = tf.add_n(style_losses)
        style_loss *= style_weight / self.num_style_layers

        # Calculate content loss
        content_losses = []
        for i in range(len(content_outputs)):
            content_out = tf.cast(content_outputs[i], tf.float32)
            content_target = tf.cast(content_targets[i], tf.float32)
            loss = tf.reduce_mean((content_out - content_target)**2)
            content_losses.append(loss)

        content_loss = tf.add_n(content_losses)
        content_loss *= content_weight / self.num_content_layers

        total_loss = style_loss + content_loss
        return total_loss

# Step 6: Initialize the model
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

print("\n=== INITIALIZING VAN GOGH STYLE TRANSFER MODEL ===")
extractor = VanGoghStyleTransfer(content_layers, style_layers)

# Calculate Van Gogh style representation
print("Calculating Van Gogh style representation...")
van_gogh_style_features = []

num_style_images = len(van_gogh_images)
for i in range(num_style_images):
    # Ensure float32 input - FIXED
    img_batch = tf.cast(tf.expand_dims(van_gogh_images[i], 0), tf.float32)
    style_outputs = extractor.vgg(img_batch)[:extractor.num_style_layers]
    style_features = [extractor.gram_matrix(style_output) for style_output in style_outputs]
    van_gogh_style_features.append(style_features)

# Average the style features
avg_style_targets = []
for layer_idx in range(len(style_layers)):
    layer_features = [features[layer_idx] for features in van_gogh_style_features]
    avg_feature = tf.reduce_mean(tf.stack(layer_features), axis=0)
    avg_feature = tf.cast(avg_feature, tf.float32)  # Ensure float32
    avg_style_targets.append(avg_feature)

print(f"✓ Van Gogh style representation calculated from {num_style_images} images!")

# Step 7: Style transfer function - FIXED VERSION
def stylize_image(content_image, style_targets, num_iterations=500):
    """Apply Van Gogh style to content image - FIXED VERSION"""
    # Ensure float32 throughout - CRITICAL FIX
    content_image = tf.cast(content_image, tf.float32)
    content_image = tf.expand_dims(content_image, 0)

    # Get content targets
    content_targets = extractor.vgg(content_image)[extractor.num_style_layers:]
    content_targets = [tf.cast(target, tf.float32) for target in content_targets]

    # Initialize stylized image
    stylized_image = tf.Variable(content_image, dtype=tf.float32)

    opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    @tf.function()
    def train_step():
        with tf.GradientTape() as tape:
            # Ensure float32 input to VGG
            vgg_input = tf.cast(stylized_image, tf.float32)
            outputs = extractor.vgg(vgg_input)

            # Style targets are already Gram matrices, pass them directly
            loss = extractor.style_content_loss(outputs, style_targets, content_targets)

        grad = tape.gradient(loss, stylized_image)
        opt.apply_gradients([(grad, stylized_image)])
        stylized_image.assign(tf.clip_by_value(stylized_image, 0.0, 1.0))
        return loss

    print(f"Applying Van Gogh style ({num_iterations} iterations)...")
    for i in range(num_iterations):
        loss = train_step()
        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss:.4f}")

    return stylized_image[0].numpy()

# Step 8: Demo with a sample content image
print("\n=== CREATING DEMO CONTENT IMAGE ===")

def create_demo_content():
    """Create a simple demo content image"""
    img = np.zeros((400, 400, 3), dtype=np.float32)  # Ensure float32

    # Sky (blue gradient)
    for y in range(200):
        img[y, :, 2] = 0.7 - y * 0.3 / 200
        img[y, :, 0] = 0.1 + y * 0.2 / 200

    # Ground (green)
    for y in range(200, 300):
        img[y, :, 1] = 0.5 + (y-200) * 0.3 / 100
        img[y, :, 0] = 0.2

    # Tree trunk (brown)
    img[150:300, 180:220, :] = [0.4, 0.2, 0.1]

    # Tree crown (green circle)
    center_y, center_x = 130, 200
    radius = 50
    y, x = np.ogrid[:400, :400]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    img[mask] = [0.1, 0.6, 0.1]

    # Sun (yellow circle)
    center_y, center_x = 80, 320
    radius = 30
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    img[mask] = [0.9, 0.9, 0.2]

    return img

demo_content = create_demo_content()

# Resize to match style images
demo_content = np.array(Image.fromarray((demo_content * 255).astype(np.uint8)).resize((256, 256), Image.LANCZOS), dtype=np.float32) / 255.0

print("✓ Demo content image created!")

# Apply Van Gogh style
print("\n=== APPLYING VAN GOGH STYLE ===")
stylized_result = stylize_image(demo_content, avg_style_targets, num_iterations=500)

# Display results
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Original content
axes[0].imshow(demo_content)
axes[0].set_title("Original Content")
axes[0].axis('off')

# Style examples
axes[1].imshow(van_gogh_images[0])
axes[1].set_title(f"Van Gogh Style\n{image_info[0]['class']}")
axes[1].axis('off')

axes[2].imshow(van_gogh_images[1] if len(van_gogh_images) > 1 else van_gogh_images[0])
axes[2].set_title(f"Van Gogh Style\n{image_info[1]['class'] if len(image_info) > 1 else image_info[0]['class']}")
axes[2].axis('off')

# Stylized result
axes[3].imshow(stylized_result)
axes[3].set_title("Stylized Result")
axes[3].axis('off')

plt.tight_layout()
plt.show()
