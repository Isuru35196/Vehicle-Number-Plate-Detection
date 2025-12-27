import cv2
import matplotlib.pyplot as plt
import os

# Paths
input_path = "vehicle_dataset/car2.jpg"  # Change as needed
output_dir = "processed_outputs"

# Create output folder if not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load and resize image
image = cv2.imread(input_path)
image = cv2.resize(image, (600, int(image.shape[0] * 600 / image.shape[1])))

# Save original
cv2.imwrite(os.path.join(output_dir, "01_original.jpg"), image)

# Step 1: Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(output_dir, "02_grayscale.jpg"), gray)

# Step 2: Bilateral filter + Canny
filtered = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(filtered, 30, 200)
cv2.imwrite(os.path.join(output_dir, "03_canny_edges.jpg"), edged)

# Step 3: Find contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

plate_crop = None

# Step 4: Detect plate and crop
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        if w > 100 and h > 30:
            plate_crop = image[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(output_dir, "04_cropped_plate.jpg"), plate_crop)
            break

# Step 5: Show all outputs using matplotlib
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

# Original
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image")
axes[0].axis('off')

# Grayscale
axes[1].imshow(gray, cmap='gray')
axes[1].set_title("Grayscale Image")
axes[1].axis('off')

# Canny
axes[2].imshow(edged, cmap='gray')
axes[2].set_title("Canny Edge Output")
axes[2].axis('off')

# Cropped Plate
if plate_crop is not None:
    axes[3].imshow(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
    axes[3].set_title("Cropped Number Plate")
else:
    axes[3].text(0.5, 0.5, 'Plate Not Found', fontsize=14, ha='center')
axes[3].axis('off')

# Save the final figure
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "05_all_outputs_combined.png"))
plt.show()
