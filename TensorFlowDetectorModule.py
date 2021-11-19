import cv2
import numpy as np
import os
import tensorflow as tf
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


class TensorFlowDetector:
    def __init__(self, path_to_model_dir="exported-models/my_model/", model_name="saved_model",
                 path_to_labels="annotations/label_map.pbtxt", gpu_flag=True):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
        tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)

        if gpu_flag:
            # Enable GPU dynamic memory allocation
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        print('Loading model...', end='')
        start_time = time.perf_counter()

        # Load saved model and build the detection function
        self.detect_fn = tf.saved_model.load(f'{path_to_model_dir}{model_name}')

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print('Done! Took {} seconds'.format(elapsed_time))

        self.category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)

    def detect(self, img, nms=True, threshold=0.5, nms_threshold=0.2):
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(img)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        detections = self.detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        boxes = detections['detection_boxes']
        classes = detections['detection_classes']
        scores = detections['detection_scores']

        if nms:
            scores = list(np.array(scores).reshape(1, -1)[0])
            scores = list(map(float, scores))
            indices = cv2.dnn.NMSBoxes(boxes, scores, threshold, nms_threshold)

            boxes = np.asarray([box for i, box in enumerate(boxes) if i in indices])
            classes = np.asarray([label for i, label in enumerate(classes) if i in indices]).astype(np.int64)
            scores = np.asarray([score for i, score in enumerate(scores) if i in indices])

        return boxes, classes, scores

    def draw(self, img, boxes, classes, scores):
        viz_utils.visualize_boxes_and_labels_on_image_array(
            img,
            boxes,
            classes,
            scores,
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)

        return img

    @staticmethod
    def flip_image(img):
        return np.fliplr(img).copy()

    @staticmethod
    def image_to_gray(img):
        return np.tile(np.mean(img, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    def get_labels(self, classes):
        labels = []
        for i in classes:
            labels.append(self.category_index[i]['name'])
        return labels

    def find_object(self, img, label, boxes, classes):
        class_names = self.get_labels(classes)
        width, height, _ = img.shape
        found = []
        for i, box in enumerate(boxes):
            if class_names[i] == label:
                y_min = box[0] * height
                x_min = box[1] * width
                y_max = box[2] * height
                x_max = box[3] * width
                center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
                found.append({'bbox': [x_min, y_min, x_max, y_max], 'center': center})

        return found


if __name__ == "__main__":
    tf_detector = TensorFlowDetector()

    IMAGE_PATH = "images/test/"
    files = os.listdir(IMAGE_PATH)

    for file in files:
        if file[-3:] == "jpg":
            image_path = f'{IMAGE_PATH}{file}'
            print('Running inference for {}... '.format(image_path), end='')

            image = cv2.imread(image_path)

            # Convert image to grayscale
            image_gray = tf_detector.image_to_gray(image)

            # Find matches
            boxes, classes, scores = tf_detector.detect(image_gray)

            # Get location of objects with the specified label
            found_objects = tf_detector.find_object(image, "explosives1.1", boxes, classes)

            # Draw bounding boxes
            image = tf_detector.draw(image, boxes, classes, scores)

            cv2.imshow("Image", image)
            print('Done')
            while cv2.waitKey(0) & 0xff != ord('c'):
                pass

    cv2.destroyAllWindows()
