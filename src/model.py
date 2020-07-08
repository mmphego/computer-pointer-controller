import argparse
import os
import sys
import time
import subprocess
import logging

from pathlib import Path

import cv2
import numpy as np

from openvino.inference_engine import IENetwork, IECore


logger = logging.getLogger(__name__)


class Base:
    """Base Class"""

    def __init__(self, model_name, device="CPU", threshold=0.60, extensions=None):
        self.model_weights = f"{model_name}.bin"
        self.model_structure = f"{model_name}.xml"
        assert (
            Path(self.model_weights).absolute().exists()
            and Path(self.model_structure).absolute().exists()
        )

        self.device = device
        self.threshold = threshold
        self._model_size = os.stat(self.model_weights).st_size / 1024.0 ** 2

        self._ie_core = IECore()
        self.model = self._get_model()

        # Get the input layer
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
        self._init_image_w = None
        self._init_image_h = None

    def _get_model(self):
        """Helper function for reading the network."""
        try:
            try:
                model = self._ie_core.read_network(
                    model=self.model_structure, weights=self.model_weights
                )
            except AttributeError:
                logger.warn("Using an old version of OpenVINO, consider updating it!")
                model = IENetwork(
                    model=self.model_structure, weights=self.model_weights
                )
        except Exception:
            raise ValueError(
                "Could not Initialise the network. "
                "Have you entered the correct model path?"
            )
        else:
            return model

    def load_model(self):
        """Load the model into the plugin"""
        self.exec_network = self._ie_core.load_network(
            network=self.model, device_name=self.device
        )

    def predict(self, image, request_id=0):
        if not isinstance(image, np.ndarray):
            raise IOError("Image not parsed correctly.")

        p_image = self.preprocess_input(image)
        self.exec_network.start_async(
            request_id=request_id, inputs={self.input_name: p_image}
        )
        status = self.exec_network.requests[request_id].wait(-1)
        if status == 0:
            result = self.exec_network.requests[request_id].outputs[self.output_name]
            return self.draw_outputs(result, image)

    def draw_outputs(self, inference_blob, image):
        """Draw bounding boxes onto the frame."""
        if not (self._init_image_w and self._init_image_h):
            raise RuntimeError("Initial image width and height cannot be None.")
        label = "Person"
        bbox_color = (0, 255, 0)
        padding_size = (0.05, 0.25)
        text_color = (255, 255, 255)
        text_scale = 1.5
        text_thickness = 1

        coords = []
        for box in inference_blob[0][0]:  # Output shape is 1x1xNx7
            conf = box[2]
            if conf >= self.threshold:
                xmin = int(box[3] * self._init_image_w)
                ymin = int(box[4] * self._init_image_h)
                xmax = int(box[5] * self._init_image_w)
                ymax = int(box[6] * self._init_image_h)
                coords.append((xmin, ymin, xmax, ymax))

                cv2.rectangle(
                    image, (xmin, ymin), (xmax, ymax,), color=bbox_color, thickness=2,
                )

                ((label_width, label_height), _) = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_PLAIN,
                    fontScale=text_scale,
                    thickness=text_thickness,
                )

                cv2.rectangle(
                    image,
                    (xmin, ymin),
                    (
                        int(xmin + label_width + label_width * padding_size[0]),
                        int(ymin + label_height + label_height * padding_size[1]),
                    ),
                    color=bbox_color,
                    thickness=cv2.FILLED,
                )
                cv2.putText(
                    image,
                    label,
                    org=(
                        xmin,
                        int(ymin + label_height + label_height * padding_size[1]),
                    ),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=text_scale,
                    color=text_color,
                    thickness=text_thickness,
                )

        return coords, image

    def preprocess_input(self, image):
        """Helper function for processing frame"""
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        # Change data layout from HWC to CHW
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame


class Face_Detection(Base):
    """Class for the Face Detection Model."""
    def __init__(self, model_name, device="CPU", threshold=0.60, extensions=None):
        super().__init__(model_name, device="CPU", threshold=0.60, extensions=None)


class Head_Pose_Estimation(Base):
    """Class for the Head Pose Estimation Model."""
    def __init__(self, model_name, device="CPU", threshold=0.60, extensions=None):
        super().__init__(model_name, device="CPU", threshold=0.60, extensions=None)


class Facial_Landmarks(Base):
    """Class for the Facial Landmarks Detection Model."""
    def __init__(self, model_name, device="CPU", threshold=0.60, extensions=None):
        super().__init__(model_name, device="CPU", threshold=0.60, extensions=None)


class Gaze_Estimation(Base):
    """Class for the Gaze Estimation Detection Model."""
    def __init__(self, model_name, device="CPU", threshold=0.60, extensions=None):
        super().__init__(model_name, device="CPU", threshold=0.60, extensions=None)
