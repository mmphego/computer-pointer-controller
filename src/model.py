import abc
import argparse
import os
import sys
import time
import subprocess

from pathlib import Path

import cv2
import numpy as np

from loguru import logger
from openvino.inference_engine import IENetwork, IECore


__all__ = [
    "Face_Detection",
    "Head_Pose_Estimation",
    "Facial_Landmarks",
    "Gaze_Estimation",
]


class Base(abc.ABC):
    """Model Base Class"""

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
        self.exec_network = None
        self.load_model()

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
        if self.exec_network is None:
            start_time = time.time()
            self.exec_network = self._ie_core.load_network(
                network=self.model, device_name=self.device
            )
            self._model_load_time = (time.time() - start_time) * 1000
            logger.info(
                f"Model: {self.model_structure} took {self._model_load_time:.3f} ms to load."
            )

    def predict(self, image, request_id=0, draw=False):
        if not isinstance(image, np.ndarray):
            raise IOError("Image not parsed correctly.")

        p_image = self.preprocess_input(image)
        self.exec_network.start_async(
            request_id=request_id, inputs={self.input_name: p_image}
        )
        status = self.exec_network.requests[request_id].wait(-1)
        if status == 0:
            predict_start_time = time.time()
            pred_result = self.exec_network.requests[request_id].outputs[
                self.output_name
            ]
            predict_end_time = (time.time() - predict_start_time) * 1000
            if draw:
                self.preprocess_output(pred_result, image, show_bbox=draw)
            return (predict_end_time, pred_result)

    @abc.abstractmethod
    def preprocess_output(self, inference_results, image, show_bbox=False):
        """Draw bounding boxes onto the frame."""
        pass

    @staticmethod
    @abc.abstractmethod
    def draw_output(image,):
        pass

    def add_text(self, text, frame, position, font_size=0.75, color=(255, 255, 255)):
        cv2.putText(
            frame, text, position, cv2.FONT_HERSHEY_COMPLEX, font_size, color, 1,
        )

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

    def preprocess_output(self, inference_results, image, show_bbox=False):
        """Draw bounding boxes onto the frame."""
        if not (self._init_image_w and self._init_image_h):
            raise RuntimeError("Initial image width and height cannot be None.")

        coords = []
        for box in inference_results[0][0]:  # Output shape is 1x1xNx7
            conf = box[2]
            if conf >= self.threshold:
                xmin = int(box[3] * self._init_image_w)
                ymin = int(box[4] * self._init_image_h)
                xmax = int(box[5] * self._init_image_w)
                ymax = int(box[6] * self._init_image_h)
                coords.append((xmin, ymin, xmax, ymax))
                if show_bbox:
                    self.draw_output(image, xmin, ymin, xmax, ymax)
        return coords, image

    @staticmethod
    def draw_output(
        image,
        xmin,
        ymin,
        xmax,
        ymax,
        label="Person's Face",
        bbox_color=(0, 255, 0),
        padding_size=(0.05, 0.25),
        text_color=(255, 255, 255),
        text_scale=2,
        text_thickness=2,
    ):

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
            org=(xmin, int(ymin + label_height + label_height * padding_size[1]),),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=text_scale,
            color=text_color,
            thickness=text_thickness,
        )


class Head_Pose_Estimation(Base):
    """Class for the Head Pose Estimation Model."""

    def __init__(self, model_name, device="CPU", threshold=0.60, extensions=None):
        super().__init__(model_name, device="CPU", threshold=0.60, extensions=None)

    def preprocess_output(self, inference_results, image):
        pass

    def draw_output(coords, image):
        pass


class Facial_Landmarks(Base):
    """Class for the Facial Landmarks Detection Model."""

    def __init__(self, model_name, device="CPU", threshold=0.60, extensions=None):
        super().__init__(model_name, device="CPU", threshold=0.60, extensions=None)

    def preprocess_output(self, inference_results, image):
        pass

    def draw_output(coords, image):
        pass


class Gaze_Estimation(Base):
    """Class for the Gaze Estimation Detection Model."""

    def __init__(self, model_name, device="CPU", threshold=0.60, extensions=None):
        super().__init__(model_name, device="CPU", threshold=0.60, extensions=None)

    def preprocess_output(self, inference_results, image):
        pass

    def draw_output(coords, image):
        pass
