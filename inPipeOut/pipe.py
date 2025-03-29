import cv2
import os
import glob
import zipfile
import tempfile

"""
Comment Color Representation:
* Aim of the function
? Input information for the function
! Any particular things to keep in mind while using the function.
"""


def gaussian_splat(frame):
    """
    * Dummy Gaussian splatting function. 
    ! Replace this with actual Gaussian splatting implementation.
    ? here i just used a simple gaussian blur.
    """
    
    # Apply a Gaussian blur as a placeholder
    return cv2.GaussianBlur(frame, (5, 5), 0)


def extract_frames(video_path, target_fps):
    """
    * Extract frames from a video file at a specified target FPS.

    ? Args:
        video_path (str): Path to the video file.
        target_fps (float): Desired frames per second to extract.

    !Yields:
        Processed frame (after Gaussian splatting) if processing is desired, otherwise yield the raw frame.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    # * Get source FPS; if unavailable, assume 30 FPS.
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0

    # * Calculate frame interval (skip frames if source FPS is higher than target FPS)
    frame_interval = max(1, int(round(source_fps / target_fps)))
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            splatted_frame = gaussian_splat(frame)
            yield splatted_frame

        count += 1

    cap.release()


def process_video(video_path, target_fps):
    # * Process a single video file.

    yield from extract_frames(video_path, target_fps)


def process_folder(folder_path, target_fps):
    """
    * Process all video files in a folder.

    ! The glob pattern can be adjusted based on the file types.
    """

    video_files = glob.glob(os.path.join(folder_path, "*"))
    for video_file in video_files:
        print(f"Processing video: {video_file}")
        yield from process_video(video_file, target_fps)


def process_zip(zip_path, target_fps):
    # * Extract a zip file to a temporary directory and process contained videos.
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        yield from process_folder(tmpdir, target_fps)


def process_live_stream(source, target_fps):
    """
    * Process a live stream.

    ? Args:
        source: An integer (e.g., 0 for the default webcam) or a stream URL.
        target_fps (float): The desired frame extraction rate.
    """

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error opening live stream: {source}")
        return

    # * Attempt to get the live stream's FPS; otherwise, assume 30 FPS.
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0
    frame_interval = max(1, int(round(source_fps / target_fps)))
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            yield gaussian_splat(frame)

        count += 1

    cap.release()


def input_pipeline(input_source, input_type, target_fps=5):
    """
    * Main pipeline function to handle different input types.

    ? Args:
        input_source (str or int): The path to the input or a live stream source.
        input_type (str): One of "video", "folder", "zip", or "stream".
        target_fps (float): The target FPS for frame extraction.

    ! Returns:
        A generator yielding processed frames.
    """

    if input_type == "video":
        return process_video(input_source, target_fps)
    elif input_type == "folder":
        return process_folder(input_source, target_fps)
    elif input_type == "zip":
        return process_zip(input_source, target_fps)
    elif input_type == "stream":
        return process_live_stream(input_source, target_fps)
    else:
        raise ValueError(
            "Unsupported input type. Use 'video', 'folder', 'zip', or 'stream'."
        )


if __name__ == "__main__":
    # import os

    print("Script started...")

    source_path = r"D:\Guassian_project\Gaussian-Splat\test_data\1.mp4"
    input_type = "video"
    target_fps = 5

    print("Checking file:", source_path)
    print("File exists:", os.path.exists(source_path))

    # * Create the frame generator from the input pipeline.
    frame_generator = input_pipeline(source_path, input_type, target_fps)

    # * Create an output directory to save the frames.
    output_dir = r"output_frames"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory already exists: {output_dir}")

    frame_count = 0
    for idx, frame in enumerate(frame_generator):
        print(f"Processed frame {idx}")

        # * Save the frame as a PNG image.
        frame_filename = os.path.join(output_dir, f"frame_{idx:04d}.png")
        cv2.imwrite(frame_filename, frame)

        cv2.imshow("Processed Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    cv2.destroyAllWindows()
    print(f"Total frames processed and saved: {frame_count}")
