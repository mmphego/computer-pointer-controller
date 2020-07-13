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
            --facial-landmarks-model models/landmarks-regression-retail-0009 \
            --head-pose-model models/head-pose-estimation-adas-0001 \
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
    video_feed = InputFeeder(input_file=args.input)

    face_detection = Face_Detection(
        model_name=args.face_model,
        source_width=video_feed.source_width,
        source_height=video_feed.source_height,
        device=args.device,
        threshold=args.prob_threshold,
    )
    facial_landmarks = Facial_Landmarks(args.facial_landmarks_model, device=args.device)
    head_pose_estimation = Head_Pose_Estimation(
        args.head_pose_model, device=args.device
    )
    gaze_estimation = Gaze_Estimation(args.gaze_model, device=args.device)

    model_load_time = (
        face_detection._model_load_time
        + head_pose_estimation._model_load_time
        + facial_landmarks._model_load_time
        + gaze_estimation._model_load_time
    ) / 1000
    logger.info(f"Total time taken to load all the models: {model_load_time:.2f} secs.")

    for frame in video_feed.next_frame():
        if args.debug:
            video_feed.show(video_feed.resize(frame))

        predict_end_time, face_bboxes = face_detection.predict(frame, draw=True)
        text = f"Face Detection Inference time: {predict_end_time:.3f} s"
        face_detection.add_text(text, frame, (15, video_feed.source_height - 80))

        if face_bboxes:
            for face_bbox in face_bboxes:
                # Useful resource: https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/

                # Face bounding box coordinates cropped from the face detection inference
                # are face_bboxes i.e `xmin, ymin, xmax, ymax`
                # Therefore the face can be cropped by:
                # frame[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]

                # extract the face ROI
                (x, y, w, h) = face_bbox
                face = frame[y:h, x:w]
                (face_height, face_width) = face.shape[:2]
                #  video_feed.show(frame[y:h, x:w], "face")

                # ensure the face width and height are sufficiently large
                if face_height < 20 or face_width < 20:
                    continue

                predict_end_time, eyes_coords = facial_landmarks.predict(face, draw=True)
                text = f"Facial Landmarks Est. Inference time: {predict_end_time:.3f} s"
                facial_landmarks.add_text(
                    text, frame, (15, video_feed.source_height - 60)
                )

    video_feed.close()


if __name__ == "__main__":
    args = arg_parser()
    main(args)
