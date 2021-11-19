import os
import cv2
from TensorFlowDetectorModule import TensorFlowDetector

tf_detector = TensorFlowDetector()

IMAGE_PATH = "images/test/"
files = os.listdir(IMAGE_PATH)

for file in files:
    if file[-3:] == "jpg":
        image_path = f'{IMAGE_PATH}{file}'
        print('Running inference for {}... '.format(image_path), end='')

        img = cv2.imread(image_path)

        # Convert image to grayscale
        image_gray = tf_detector.image_to_gray(img)

        # Find matches
        boxes, classes, scores = tf_detector.detect(image_gray, threshold=0.7)

        # Get location of objects with the specified label
        found_objects = tf_detector.find_object(img, "explosives1.1", boxes, classes)

        # Draw bounding boxes
        img = tf_detector.draw(img, boxes, classes, scores)

        cv2.imshow("Image", img)
        print('Done')
        while cv2.waitKey(0) & 0xff != ord('c'):
            pass

    cv2.destroyAllWindows()
