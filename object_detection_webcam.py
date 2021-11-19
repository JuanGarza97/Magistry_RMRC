import cv2
from TensorFlowDetectorModule import TensorFlowDetector

tf_detector = TensorFlowDetector()

cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 640)

while cap.isOpened():
    success, img = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'FPS: {fps}')

    if success:
        # Convert image to grayscale
        image_gray = tf_detector.image_to_gray(img)

        # Find matches
        boxes, classes, scores = tf_detector.detect(image_gray, threshold=0.7)

        # Get location of objects with the specified label returns [{bbox, center}]
        found_objects = tf_detector.find_object(img, "explosives1.1", boxes, classes)

        # Draw bounding boxes
        img = tf_detector.draw(img, boxes, classes, scores)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
