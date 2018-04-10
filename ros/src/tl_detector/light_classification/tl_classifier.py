from styx_msgs.msg import TrafficLight
import cv2
import random
import time
import tensorflow as tf
import os
import numpy as np

class TLClassifier(object):
    def __init__(self):
        print("TF Detector init")
        CWD = os.path.dirname(os.path.realpath(__file__))
        PATH_TO_CKPT = CWD + '/../../../../data/exported_graphs/frozen_inference_graph.pb'
        PATH_TO_LABELS = '/../../../../data/label_map.pbtxt'
        # Size, in inches, of the output images.
        IMAGE_SIZE = (12, 8)
        NUM_CLASSES = 3
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            print("Importing")
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                print("Imported")
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        #label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        print("get classification")
        (im_height, im_width) = image.shape[:2]
        image_np = image.reshape(
          (im_height, im_width, 3)).astype(np.uint8)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        print("running infer")
        output_dict = self.run_inference_for_single_image(image_np, self.detection_graph)
        detected_boxes = []
        i = 0
        for score in output_dict["detection_scores"]:
            if score > 0.7:
                detected_boxes.append({
                    'score': score,
                    'bounding_box': output_dict["detection_boxes"][i],
                    'detection_class': output_dict["detection_classes"][i]
                        })
            i = i + 1
        print(detected_boxes)
        return TrafficLight.UNKNOWN

    def run_inference_for_single_image(self, image, graph):
        print("starting run infer")
        with graph.as_default():
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            print("running session")
            output_dict = self.sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})
            print("session done")

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

