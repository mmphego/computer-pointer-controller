import argparse

from src import input_feeder, model, mouse_controller


def arg_parser():
    """Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--face-model",
        required=True,
        type=str,
        help="Path to an xml file with a trained model.",
    )
    parser.add_argument(
        "--head-pose-model",
        required=True,
        type=str,
        help="Path to an IR model representative for head-pose-model"
        )
    parser.add_argument(
        "--facial-landmarks-model",
        required=True,
        type=str,
        help="Path to an IR model representative for facial-landmarks-model"
        )
    parser.add_argument(
        "--gaze-model",
        required=True,
        type=str,
        help="Path to an IR model representative for gaze-model"
        )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to image or video file or 'cam' for Webcam.",
    )

    parser.add_argument(
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
        "--out", action="store_true", help="Write video to file.",
    )
    parser.add_argument(
        "--ffmpeg", action="store_true", help="Flush video to FFMPEG.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show output on screen [debugging].",
    )

    return parser.parse_args()


def main(args):
    pass


if __name__ == '__main__':
    args = arg_parser()
    main(args)
