"""
USAGE

xhost +; docker run --rm -ti \
--volume "$PWD":/app \
--env DISPLAY=$DISPLAY \
--volume="$HOME/.Xauthority":/root/.Xauthority \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--device /dev/video0 \
mmphego/intel-openvino \
    bash -c "source /opt/intel/openvino/bin/setupvars.sh && \
        python main.py \
            --face-model models/face-detection-adas-binary-0001 \
            --head-pose-model models/head-pose-estimation-adas-0001 \
            --facial-landmarks-model models/face-detection-adas-binary-0001 \
            --gaze-model models/gaze-estimation-adas-0002 \
            --input resources/demo.mp4";
"""

import argparse
from loguru import logger

from src.input_feeder import InputFeeder
from src.model import (
    Facial_Landmarks,
    Face_Detection,
    Head_Pose_Estimation,
    Gaze_Estimation,
)

from src.mouse_controller import MouseController


def arg_parser():
    """Parse command line arguments.

    :return: command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-fm",
        "--face-model",
        required=True,
        type=str,
        help="Path to an xml file with a trained model.",
    )
    parser.add_argument(
        "-hp",
        "--head-pose-model",
        required=True,
        type=str,
        help="Path to an IR model representative for head-pose-model",
    )
    parser.add_argument(
        "-fl",
        "--facial-landmarks-model",
        required=True,
        type=str,
        help="Path to an IR model representative for facial-landmarks-model",
    )
    parser.add_argument(
        "-gm",
        "--gaze-model",
        required=True,
        type=str,
        help="Path to an IR model representative for gaze-model",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="CPU",
        help="Specify the target device to infer on: "
        "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
        "will look for a suitable plugin for device "
        "specified (CPU by default)",
    )
    parser.add_argument(
        "-pt",
        "--prob_threshold",
        type=float,
        default=0.8,
        help="Probability threshold for detections filtering" "(0.8 by default)",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Path to image or video file or 'cam' for Webcam.",
    )
    parser.add_argument(
        "--out", action="store_true", help="Write video to file.",
    )
    parser.add_argument(
        "-mp",
        "--mouse-precision",
        type=str,
        default="low",
        help="The precision for mouse movement (how much the mouse moves).",
    )
    parser.add_argument(
        "-ms",
        "--mouse-speed",
        type=str,
        default="fast",
        help="The speed (how fast it moves) by changing",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show output on screen [debugging].",
    )

    return parser.parse_args()


def main(args):
    mouse_controller = MouseController(
        precision=args.mouse_precision, speed=args.mouse_speed
    )
    face_detection = Face_Detection(
        args.face_model, device=args.device, threshold=args.prob_threshold
    )
    head_pose_estimation = Head_Pose_Estimation(
        args.head_pose_model, device=args.device
    )
    facial_landmarks = Facial_Landmarks(args.facial_landmarks_model, device=args.device)
    gaze_estimation = Gaze_Estimation(args.gaze_model, device=args.device)

    model_load_time = (
        face_detection._model_load_time
        + head_pose_estimation._model_load_time
        + facial_landmarks._model_load_time
        + gaze_estimation._model_load_time
    ) / 1000
    logger.info(f"Total time taken to load all the models: {model_load_time:.2f} secs.")

    video_feed = InputFeeder(input_file=args.input)

    for frame in video_feed.next_frame():
        video_feed.show(frame)
    video_feed.close()


if __name__ == "__main__":
    args = arg_parser()
    main(args)
