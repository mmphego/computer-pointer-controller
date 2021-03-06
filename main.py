#!/usr/bin/env python3

import argparse
import time

from loguru import logger
from pprint import pprint

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
        "specified (Default: CPU)",
    )
    parser.add_argument(
        "-pt",
        "--prob_threshold",
        type=float,
        default=0.8,
        help="Probability threshold for detections filtering" "(Default: 0.8)",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Path to image,  video file or 'cam' for Webcam.",
    )
    parser.add_argument(
        "--out", action="store_true", help="Write video to file.",
    )
    parser.add_argument(
        "-mp",
        "--mouse-precision",
        type=str,
        default="low",
        const="low",
        nargs="?",
        choices=["high", "low", "medium"],
        help="The precision for mouse movement (how much the mouse moves). [Default: low]",
    )
    parser.add_argument(
        "-ms",
        "--mouse-speed",
        type=str,
        default="fast",
        const="fast",
        nargs="?",
        choices=["fast", "slow", "medium"],
        help="The speed (how fast it moves) by changing [Default: fast]",
    )
    parser.add_argument(
        "--enable-mouse", action="store_true", help="Enable Mouse Movement",
    )
    parser.add_argument(
        "--show-bbox",
        action="store_true",
        help="Show bounding box and stats on screen [debugging].",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show output on screen [debugging].",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Verbose OpenVINO layer performance stats [debugging].",
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
    count = 0
    for frame in video_feed.next_frame():
        count += 1
        predict_end_time, face_bboxes = face_detection.predict(
            frame, show_bbox=args.show_bbox
        )

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
                # video_feed.show(frame[y:h, x:w], "face")

                # ensure the face width and height are sufficiently large
                if face_height < 20 or face_width < 20:
                    continue

                facial_landmarks_pred_time, eyes_coords = facial_landmarks.predict(
                    face, show_bbox=args.show_bbox
                )

                hp_est_pred_time, head_pose_angles = head_pose_estimation.predict(
                    face, show_bbox=args.show_bbox
                )

                gaze_pred_time, gaze_vector = gaze_estimation.predict(
                    frame,
                    show_bbox=args.show_bbox,
                    face=face,
                    eyes_coords=eyes_coords,
                    head_pose_angles=head_pose_angles,
                )
                if args.debug:
                    head_pose_estimation.show_text(frame, head_pose_angles)
                    gaze_estimation.show_text(frame, gaze_vector)

                if args.enable_mouse:
                    mouse_controller.move(gaze_vector["x"], gaze_vector["y"])
        else:
            if count % 10 ==0:
                logger.warning("Could not detect face in the frame.")

        if args.debug:
            if face_bboxes:
                text = f"Face Detection Inference time: {predict_end_time:.3f} s"
                face_detection.add_text(
                    text, frame, (15, video_feed.source_height - 80)
                )
                text = (
                    f"Facial Landmarks Est. Inference time: "
                    f"{facial_landmarks_pred_time:.3f} s"
                )
                facial_landmarks.add_text(
                    text, frame, (15, video_feed.source_height - 60)
                )
                text = f"Head Pose Est. Inference time: {hp_est_pred_time:.3f} s"
                head_pose_estimation.add_text(
                    text, frame, (15, video_feed.source_height - 40)
                )
                text = f"Gaze Est. Inference time: {gaze_pred_time:.3f} s"
                gaze_estimation.add_text(
                    text, frame, (15, video_feed.source_height - 20)
                )
            video_feed.show(video_feed.resize(frame))

        if args.stats:
            stats = {
                "face_detection": face_detection.perf_stats,
                "facial_landmarks": facial_landmarks.perf_stats,
                "head_pose_estimation": head_pose_estimation.perf_stats,
                "gaze_estimation": gaze_estimation.perf_stats,
            }
            pprint(stats)

    video_feed.close()


if __name__ == "__main__":
    args = arg_parser()
    main(args)
