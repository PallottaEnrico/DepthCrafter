import numpy as np
import cv2

dataset_res_dict = {
    "sintel":[448, 1024],
    "scannet":[640, 832],
    "kitti":[384, 1280],
    "bonn":[512, 640],
    "nyu":[448, 640],
}

h = 240
half = 40

def read_video_frames(video_path, process_length, target_fps, max_res, dataset):
    # a simple function to read video frames
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # round the height and width to the nearest multiple of 64

    if dataset=="open":        
        height = round(original_height / 64) * 64
        width = round(original_width / 64) * 64
    else:
        height = dataset_res_dict[dataset][0]
        width = dataset_res_dict[dataset][1]
    
    # resize the video if the height or width is larger than max_res
    if max(height, width) > max_res:
        scale = max_res / max(original_height, original_width)
        height = round(original_height * scale / 64) * 64
        width = round(original_width * scale / 64) * 64

    if target_fps < 0:
        target_fps = original_fps

    stride = max(round(original_fps / target_fps), 1)

    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (process_length > 0 and frame_count >= process_length):
            break
        if frame_count % stride == 0:
            # First do UCF preprocessing
            frame = frame[:, half:half+h, :]
            frame = cv2.resize(frame, (256, 256))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frames.append(frame.astype("float32") / 255.0)
        frame_count += 1
    cap.release()

    frames = np.array(frames)
    return frames, target_fps


def save_video(
    video_frames,
    output_video_path,
    fps: int = 15,
) -> str:
    # a simple function to save video frames
    height, width = video_frames[0].shape[:2]
    is_color = video_frames[0].ndim == 3
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps, (width, height), isColor=is_color
    )

    for frame in video_frames:
        frame = (frame * 255).astype(np.uint8)
        if is_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)

    video_writer.release()
    return output_video_path
