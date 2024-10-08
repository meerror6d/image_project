import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

def resize_and_pad(image, dimentions):
    """ Resize and pad images to a fixed size. """
    h, w = image.shape[:2]
    sh, sw = dimentions

    # Interpolation method
    if h > sh or w > sw:  # Shrinking image
        interp = cv2.INTER_AREA
    else:  # Stretching image
        interp = cv2.INTER_CUBIC

    # Aspect ratio of image
    aspect = w/h

    # Computing scaling and pad sizing
    if aspect > 1:  # Horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)//2
        pad_top, pad_bot = pad_vert, pad_vert
        pad_left, pad_right = 0, 0  # No horizontal padding needed if image is wider
    elif aspect < 1:  # Vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)//2
        pad_left, pad_right = pad_horz, pad_horz
        pad_top, pad_bot = 0, 0  # No vertical padding needed if image is taller
    else:  # Square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # Resize and pad
    scaled_img = cv2.resize(image, (new_w, new_h), interpolation=interp)
    padded_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=[255, 255, 255])  # White padding

    return padded_img

def get_descriptor(border_image, binary_image,eroded_image):
    perimeter = np.count_nonzero(border_image)
    
    pixels = []
    a=[]
    height, width = border_image.shape

    for x in range(eroded_image.shape[0]):
        for y in range(eroded_image.shape[1]):
            if eroded_image[x,y]==0:
                a.append((x,y))
    area=len(a)

    for y in range(height):
        for x in range(width):
            if border_image[y, x] > 0:
                pixels.append((y, x))
    
    if len(pixels) == 0:
        return None, None, None
    
    x_min = y_min = float('inf')
    x_max = y_max = float('-inf')
    
    for y, x in pixels:
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y
    
    horizontal_extent = x_max - x_min
    vertical_extent = y_max - y_min
    
    max_diameter = max(horizontal_extent, vertical_extent)
    
    compactness = (perimeter * perimeter) / area if area != 0 else 0
    form_factor = (4 * np.pi * area) / (perimeter * perimeter) if perimeter != 0 else 0
    roundness = (4 * area) / (np.pi * max_diameter * max_diameter) if max_diameter != 0 else None


    print("area: ",area)
    print("perimeter: ",perimeter)
    print("max_diameter",max_diameter)
    print("compactness: ",compactness)
    print("form factor",form_factor)
    print("roundness",roundness)

    return area,perimeter,max_diameter,compactness, form_factor, roundness

def customized_threshold(image, threshold):
    binary_image = np.zeros_like(image, dtype=np.uint8)
    binary_image[image > threshold] = 255
    return binary_image

def process_image(image_path, train_descriptors):
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load the image. Please check the file path.")
        return
    
    dimensions = (image.shape[1], image.shape[0])  # (width, height)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray Image', gray_image)

    # Apply binary thresholding to create a binary image
    binary_image = customized_threshold(gray_image, 220)

    # Apply erosion to the binary image
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=2)
    cv2.imshow('Eroded Image', eroded_image)

    
    border_image = cv2.subtract(binary_image, eroded_image)
    cv2.imshow('Border Image', border_image)

    # Find connected components in the binary image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(border_image)

    labeled_image = image.copy()
    region_descriptors = {}

    def match_descriptors(test_descriptors, train_descriptors):
        min_distance = float('inf')
        best_match = None
        
        print("Testing Descriptors:", test_descriptors)
        
        for shape, descriptors in train_descriptors.items():
            distance = 0
            for i in range(len(test_descriptors)):
                distance += (test_descriptors[i] - descriptors[i]) ** 2
            distance = np.sqrt(distance)
            
            if distance < min_distance:
                min_distance = distance
                best_match = shape
        print(distance)
        print('distance:',min_distance)
        print(f"Best Match: {best_match}")
        return best_match


    # Colors for different categories
    colors = {
        'chips': (0, 255, 0),    # Green
        'can': (255, 0, 0),      # Blue
        'bottle': (0, 0, 255)    # Red
    }

    # Process each labeled region
    for label in range(1, num_labels):  # Start from 1 to skip the background
        x, y, w, h, area = stats[label]
        if area >= 300:  # Only consider significant regions
            # Get the binary mask for the current region
            region_mask = (labels == label).astype(np.uint8) * 255

            # Calculate descriptors for the current region
            area,perimeter,max_diameter,compactness, form_factor, roundness = get_descriptor(region_mask, border_image,eroded_image)

            if compactness is None or form_factor is None or roundness is None:
                continue

            # Store the descriptors
            region_descriptor = (area,perimeter,max_diameter,compactness, form_factor, roundness)

            # Match region descriptors with trained descriptors
            best_match = match_descriptors(region_descriptor, train_descriptors)

            # Determine the color based on the best match
            color = colors.get(best_match, (255, 255, 255))  # Default to white if not matched

            # Label the region with the best match
            region_label = f'{best_match} - item{label}'
            cv2.rectangle(labeled_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(labeled_image, region_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Store the descriptors for output
            region_descriptors[region_label] = {
                'area':area,
                'perimeter':perimeter,
                'max_diameter':max_diameter,
                'compactness': compactness,
                'form factor': form_factor,
                'roundness': roundness
            }

    # Convert the original image from BGR to RGB for displaying with matplotlib
    original_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    labeled_image_rgb = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)

    # Display the original and labeled images
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(labeled_image_rgb)
    plt.title('Labeled Regions: Chips, Cans, Bottles')
    plt.axis('off')

    plt.show()

    # Print the region descriptors
    print(json.dumps(region_descriptors, indent=4))

    # Save the labeled image and descriptors
    cv2.imwrite('labeled_image_with_items.jpg', labeled_image)
    with open('region_descriptors.json', 'w') as json_file:
        json.dump(region_descriptors, json_file, indent=4)

# Load or calculate train descriptors for known shapes
# Here, we're assuming you have images for 'chips', 'can', and 'bottle' to extract descriptors
def get_trained_descriptor(image_path,s):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    im=cv2.imread('one.jpg')
    dimensions=(im.shape[0],im.shape[1])
    image = resize_and_pad(image, dimensions)  # Resize the training image dynamically
    binary_image = customized_threshold(image, 235)
    eroded_image = cv2.erode(binary_image, np.ones((3, 3), np.uint8))
    border_image = cv2.subtract(binary_image, eroded_image)
    cv2.imshow('', eroded_image)
    #cv2.imshow('',border_image)
    print('item: ',s)
    return get_descriptor(border_image, binary_image,eroded_image)

# Train descriptors
train_descriptors = {
    'chips': get_trained_descriptor('chips1.jpg','chips'),  # Provide actual paths to these images
    'can': get_trained_descriptor('can1.jpg','can'),
    'bottle': get_trained_descriptor('bottle1.jpg','bottle')
}

# Process the uploaded image with multiple items
process_image('one.jpg', train_descriptors)

cv2.waitKey(0)
cv2.destroyAllWindows()