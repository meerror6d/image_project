import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the image
image_path = 'store.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize the image for easier processing
image_resized = cv2.resize(image_rgb, (800, 600))

# Convert to grayscale
gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)

# Apply GaussianBlur to reduce noise and detail in the image
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edge map
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract features from contours
features = []
bounding_boxes = []
for contour in contours:
    if cv2.contourArea(contour) > 500:  # filter out small contours
        x, y, w, h = cv2.boundingRect(contour)
        roi = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (20, 20)).flatten()
        features.append(roi_resized)
        bounding_boxes.append((x, y, w, h))

# Apply KMeans clustering to classify the features
kmeans = KMeans(n_clusters=3)
kmeans.fit(features)
labels = kmeans.labels_

# Map labels to item names (bottles, chips, cans)
label_to_item = {0: 'Item 1', 1: 'Item 2', 2: 'Item 3'}
item_counts = {label_to_item[label]: labels.tolist().count(label) for label in set(labels)}

print(f"Item counts: {item_counts}")

# Draw bounding boxes and labels on the image
output_image = image_resized.copy()
for (x, y, w, h), label in zip(bounding_boxes, labels):
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(output_image, label_to_item[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Display the result
plt.figure(figsize=(12, 8))
plt.imshow(output_image)
plt.title("Detected Items in Grocery Store")
plt.axis('off')
plt.show()
