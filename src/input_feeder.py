import mimetypes
import os

import cv2
from tqdm import tqdm

from loguru import logger
from numpy import ndarray


class FormatNotSupported(Exception):
    pass


class InputFeeder:
    def __init__(self, input_file=None):
        """
        This class can be used to feed input from an image, webcam, or video to your model.

        Parameters
        ----------
        input_file: str
            The file that contains the input image or video file.
            Leave empty for cam input_type.

        Example
        -------
        ```
            feed=InputFeeder(input_file='video.mp4')
            for batch in feed.next_frame():
                do_something(batch)
            feed.close()
        ```
        """
        self.input_file = input_file
        assert isinstance(self.input_file, str)
        try:
            self._input_type, _ = mimetypes.guess_type(self.input_file)
            assert isinstance(self._input_type, str)
        except AssertionError:
            self._input_type = ""
        self._progress_bar = None
        self.load_data()

    def load_data(self):
        if "video" in self._input_type:
            self.cap = cv2.VideoCapture(self.input_file)
        elif "image" in self._input_type:
            self.cap = cv2.imread(self.input_file)
        elif "cam" in self.input_file.lower():
            self._input_type = self.input_file
            self.cap = cv2.VideoCapture(0)
        else:
            msg = f"Source: {self.input_file} not supported!"
            logger.warn(msg)
            raise FormatNotSupported(msg)
        logger.info(f"Loaded input source type: {self._input_type}")

    @property
    def source_width(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def source_height(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def video_len(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self):
        return int(self.cap.get(cv2.CAP_PROP_FPS))

    @property
    def progress_bar(self):
        if not self._progress_bar:
            self._progress_bar = tqdm(total=int(self.video_len - self.fps + 1))
        return self._progress_bar

    def resize(self, frame, height=None, width=None):
        if (height and width) is None:
            width, height = (self.source_width//2, self.source_height//2)
        return cv2.resize(frame, (width, height))

    def show(self, frame, frame_name="video"):
        cv2.imshow(frame_name, frame)

    def write_video(self, output_path=".", filename="output_video.mp4"):
        out_video = cv2.VideoWriter(
            os.path.join(output_path, filename),
            cv2.VideoWriter_fourcc(*"avc1"),
            fps,
            (self.source_width, self.source_height),
            True,
        )
        return out_video

    def next_frame(self, quit_key="q"):
        """Returns the next image from either a video file or webcam."""
        while self.cap.isOpened():
            self.progress_bar.update(1)
            flag = False
            for _ in range(1):
                flag, frame = self.cap.read()

            if not flag:
                break
            yield frame

            key = cv2.waitKey(1) & 0xFF
            # if `quit_key` was pressed, break from the loop
            if key == ord(quit_key):
                break

    def close(self):
        """Closes the VideoCapture."""
        if "image" in self._input_type:
            self.cap.release()
        if self.progress_bar:
            self.progress_bar.close()
        cv2.destroyAllWindows()
        logger.info("============ CleanUp! ============")
