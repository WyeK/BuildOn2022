import cv2

from pixellib import semantic


def ade20k_detection():
    capture = cv2.VideoCapture("examplevid.mp4")

    segment_video = semantic.semantic_segmentation()
    segment_video.load_ade20k_model("deeplabv3_xception65_ade20k.h5")
    segment_video.process_camera_ade20k(capture, overlay=True, frames_per_second=15, show_frames=True,
                                        frame_name="frame", extract_segmented_objects=True, remote=False, output_video_name="output_video.mp4")


if __name__ == "__main__":
    ade20k_detection()
