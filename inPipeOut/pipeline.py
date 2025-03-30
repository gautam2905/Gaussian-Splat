import cv2 as cv
import os
import glob
import zipfile as zp
import tempfile as tp
import pycolmap as pc

"""
Comment Color Representation:
* Aim of the function
? Input information for the function
! Any particular things to keep in mind while using the function.
"""


class pipeline:
    def process_file(source_file, input_type, target_fps=5):
        if not (os.path.exists(source_file)):
            print("File does not exist")
            return

        frame_generator = pipeline.input_pipeline(source_file, input_type, target_fps)
        paths_list = pipeline.make_output_directory_structure()

        pipeline.save_frames(paths_list["images"], frame_generator)

        pipeline.colmap_run()

    def extract_frames(video_path, target_fps):

        # TODO: make comments.

        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"(extract_frames) Error Opening Video: {video_path}")
            return

        # * get source FPS; by default 30fps.
        source_fps = cap.get(cv.CAP_PROP_FPS)
        if source_fps <= 0:
            source_fps = 30.0

        # * calculate frame interval (skip frames if source fps is higher than target fps)
        frame_interval = max(1, int(round(source_fps / target_fps)))
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                yield frame
            count += 1

        cap.release()

    def save_frames(save_path, frame_generator):
        if not (os.path.exists(save_path)):
            os.makedirs(save_path)
            print(f"Created output directory: {save_path}")
        else:
            print(f"Output directory already exist: {save_path}")

        frame_count = 0
        for idx, frame in enumerate(frame_generator):
            print(f"Processed frame {idx}")

            file_name = os.path.join(save_path, f"frame_{idx:04d}.png")
            cv.imwrite(file_name, frame)
            cv.imshow("Processed Frame", frame)
            if (cv.waitKey(1)) & (0xFF == ord("q")):
                break
            frame_count += 1

        cv.destroyAllWindows()
        print(f"Total frames processed and saved: {frame_count}")

    def process_video(video_path, target_fps):

        # * processes a single video

        yield from pipeline.extract_frames(video_path, target_fps)

    def process_folder(folder_path, target_fps):

        # TODO: make comments

        video_files = glob.glob(os.path.join(folder_path, "*"))
        for video in video_files:
            print(f"(process_folder) Processing Video : {video}")
            yield from pipeline.process_video(video_path=video, target_fps=target_fps)

    def process_zip(zip_path, target_fps):

        # TODO : make comments
        """
        ! major assumption :
            - zip file only contains video files.
            - if not need to change the code slightly
        """

        with tp.TemporaryDirectory() as tmp_dir:
            with zp.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmp_dir)
            yield from pipeline.process_folder(tmp_dir, target_fps=target_fps)

    def process_live_stream(source, target_fps):
        print("\nProcessing_live_stream\n")
        pipeline.extract_frames(video_path=source, target_fps=target_fps)

    def input_pipeline(input_source, input_type, target_fps=5):

        # TODO : make comments

        if input_type == "video":
            return pipeline.process_video(input_source, target_fps)
        elif input_type == "folder":
            return pipeline.process_folder(input_source, target_fps)
        elif input_type == "zip":
            return pipeline.process_zip(input_source, target_fps)
        elif input_type == "stream":
            return pipeline.process_live_stream(input_source, target_fps)
        else:
            raise ValueError(
                "Unsupported input type. Use 'video', 'folder,'zip', or 'stream'"
            )

    def make_output_directory_structure() -> dict:
        """
        ? Goal: Create the required directory structure for COLMAP.

        The structure will look like this (relative to current working directory):

        ├── images         --> For saving extracted frames.
        ├── input          --> (Optional) Backup or original frames.
        ├── distorted      --> For COLMAP database and undistorted images.
        │     └── sparse/0 --> Sparse reconstruction output (cameras.bin, images.bin, points3D.bin, etc.)
        ├── sparse         --> (Optional) Another sparse output folder.
        │     └── 0      --> Final sparse reconstruction (e.g., with points3D.ply)
        └── stereo         --> For dense reconstruction outputs.
              ├── consistency_graphs
              ├── depth_maps
              └── normal_maps
        """

        base_dir = os.getcwd()
        dirs = {
            "images": os.path.join(base_dir, "images"),
            "input": os.path.join(base_dir, "input"),
            "distorted": os.path.join(base_dir, "distorted"),
            "distorted_sparse": os.path.join(base_dir, "distorted", "sparse", "0"),
            "sparse": os.path.join(base_dir, "sparse", "0"),
            "stereo": os.path.join(base_dir, "stereo"),
            "stereo_consistency": os.path.join(
                base_dir, "stereo", "consistency_graphs"
            ),
            "stereo_depth": os.path.join(base_dir, "stereo", "depth_maps"),
            "stereo_normal": os.path.join(base_dir, "stereo", "normal_maps"),
        }
        for path in dirs.values():
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")
        # Return the dictionary of paths.
        return dirs

    def colmap_run():
        """
        * Automate the COLMAP pipeline using PyCOLMAP.

        This function:
        - Runs feature extraction.
        - Runs feature matching.
        - Runs sparse reconstruction (mapping).

        Output files such as cameras.bin, images.bin, and points3D.bin are saved in the output directories.
        """

        base_dir = os.getcwd()
        # Directory where extracted frames are stored.
        image_path = os.path.join(base_dir, "images")
        # Database file location (inside 'distorted' folder).
        database_path = os.path.join(base_dir, "distorted", "database.db")
        # Output folder for sparse reconstruction.
        sparse_output = os.path.join(base_dir, "sparse", "0")

        # Ensure output directories exist (should already be created by make_output_directory_structure).
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        os.makedirs(sparse_output, exist_ok=True)

        print("Running COLMAP feature extraction...")
        # Run feature extraction using PyCOLMAP.
        pc.extract_features(database_path=database_path, image_path=image_path)

        print("Running COLMAP feature matching...")
        # Run feature matching.
        pc.match_features(database_path=database_path)

        print("Running COLMAP sparse reconstruction (mapper)...")
        # Run the mapper; this returns a reconstruction object.
        reconstruction = pc.run_mapper(
            database_path=database_path,
            image_path=image_path,
            output_path=sparse_output,
        )
        print("Sparse reconstruction complete.")

        # Optional: Inspect or log details from the reconstruction.
        print(f"Number of cameras: {len(reconstruction.cameras)}")
        print(f"Number of images: {len(reconstruction.images)}")
        print(f"Number of 3D points: {len(reconstruction.points3D)}")


if __name__ == "__main__":
    print("Script started...")
    source_path = r"D:\Guassian_project\Gaussian-Splat\test_data\1.mp4"
    input_type = "video"
    target_fps = 5

    print("Checking file:", source_path)
    print("File exists:", os.path.exists(source_path))

    # Process the file: extract frames, save them, and then run COLMAP.
    pipeline.process_file(source_path, input_type, target_fps)
