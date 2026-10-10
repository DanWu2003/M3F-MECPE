import argparse
import os
import ffmpeg
import threading
import librosa
import numpy as np
from tqdm import tqdm
import cv2

def split_list(l, n):
    """Yield successive n-sized chunks from l."""
    length = len(l)
    chunk_size = length // n + 1
    for i in range(0, length, chunk_size):
        yield l[i:i + chunk_size]

def extract_audio_from_video(video_path):
    """Use ffmpeg to extract audio from video and return audio waveform and sample rate."""
    audio_path = video_path.replace('.mp4', '.wav')
    if os.path.exists(audio_path):
        print(f"Audio file {audio_path} already exists. Loading it...")
    else:
        print(f"Extracting audio from video {video_path}...")
        ffmpeg.input(video_path).output(audio_path).run(overwrite_output=True)
    audio, sr = librosa.load(audio_path, sr=None)  # Load the extracted audio file
    return audio, sr

def get_video_duration(video_path):
    """Get the duration of the video (in seconds)."""
    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps_video
    cap.release()
    return duration, fps_video

def extract_frames_and_audio_features(video_path, output_video_dir, output_audio_dir, fps=10):
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)

    if not os.path.exists(output_audio_dir):
        os.makedirs(output_audio_dir)

    # Capture video from file
    cap = cv2.VideoCapture(video_path)
    video_duration, video_fps = get_video_duration(video_path)  # Get the original video's duration and fps
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Extract audio directly from the video
    audio, sr = extract_audio_from_video(video_path)

    # Calculate how many frames we will extract based on desired fps
    frames_needed = int(video_duration * fps)  # Total frames to extract from the video
    if frames_needed <= 0:
        raise ValueError(f"Video is too short to sample at {fps} FPS: {video_path}")
    audio_samples_needed = int(video_duration * sr)  # Total audio samples corresponding to video duration

    # Calculate audio per frame based on target fps
    audio_per_frame = audio_samples_needed / frames_needed  # Audio samples per frame
    audio_frames = [audio[int(i * audio_per_frame):int((i + 1) * audio_per_frame)] for i in range(frames_needed)]

    count = 0
    frame_id = 0
    frame_interval = max(int(video_fps / fps), 1)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Check if this frame needs to be saved
        if count % frame_interval == 0:
            if frame_id >= len(audio_frames):
                break
            # Extract audio features corresponding to the current frame
            audio_frame = audio_frames[frame_id]

            # Extract MFCC audio features for this frame
            audio_features = librosa.feature.mfcc(y=audio_frame, sr=sr, n_mfcc=13)
            print("MFCC特征的维度:", audio_features.shape)
            audio_feature_filename = f"audio_{frame_id:06d}.npy"
            audio_feature_path = os.path.join(output_audio_dir, audio_feature_filename)
            np.save(audio_feature_path, audio_features)

            # Save the corresponding frame as an image
            frame_filename = f"frame{frame_id:06d}.jpg"
            frame_path = os.path.join(output_video_dir, frame_filename)
            cv2.imwrite(frame_path, frame)

            frame_id += 1

        count += 1

    cap.release()

def extract_frames(video_name, src_base_dir, dst_base_dir, dst_base_dir_audio, fps):
    video_id = video_name.split('.')[0]
    video_output_dir = os.path.join(dst_base_dir, video_id)
    audio_output_dir = os.path.join(dst_base_dir_audio, video_id)
    if os.path.exists(video_output_dir) and os.path.exists(audio_output_dir):
        return
    video_path = os.path.join(src_base_dir, video_name)

    # Extract frames and audio features
    extract_frames_and_audio_features(
        video_path, video_output_dir, audio_output_dir, fps=fps
    )

def target(full_id_list, src_base_dir, dst_base_dir, dst_base_dir_audio, fps):
    for video_id in tqdm(full_id_list):
        try:
            extract_frames(
                video_id, src_base_dir, dst_base_dir, dst_base_dir_audio, fps
            )
        except Exception as e:
            print(f"Error processing {video_id}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract video frames and aligned MFCC features."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing videos")
    parser.add_argument("--frame-dir", required=True, help="Output directory for frames")
    parser.add_argument("--audio-dir", required=True, help="Output directory for MFCC files")
    parser.add_argument("--fps", type=int, default=10, help="Target frames per second")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker threads")
    return parser.parse_args()


def main():
    args = parse_args()
    src_base_dir = os.path.abspath(args.input_dir)
    dst_base_dir = os.path.abspath(args.frame_dir)
    dst_base_dir_audio = os.path.abspath(args.audio_dir)

    full_name_list = []
    for video_name in os.listdir(src_base_dir):
        video_path = os.path.join(src_base_dir, video_name)
        if not os.path.isfile(video_path):
            continue
        if os.path.splitext(video_name)[1].lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
            continue
        full_name_list.append(video_name)

    full_name_list.sort()
    splits = list(split_list(full_name_list, max(args.workers, 1)))

    threads = []
    for i, split in enumerate(splits):
        thread = threading.Thread(
            target=target,
            args=(
                split,
                src_base_dir,
                dst_base_dir,
                dst_base_dir_audio,
                args.fps,
            ),
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


if __name__ == '__main__':
    main()
