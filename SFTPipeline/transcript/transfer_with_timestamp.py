"""
transfer_with_timestamp.py

Use Whisper-large-v3 Model to transcribe videos under the BASIC_DATA_PATH, and save results as a .txt text file with timestamps and a JSON file.
Consider each 20s-long segment of video as a basic unit of transcription.
"""
import os
import numpy as np
import torch
import whisper
import subprocess
from tqdm import tqdm
import functools
import json
import cv2
import librosa
from typing import List, Dict, Tuple

# --- Configuration ---
MODEL_NAME = "large"
BASIC_DATA_PATH = '/thunlp_train/datasets/external-data-youtube/video/20241103/'
OUTPUT_DIR = '/user/majunxian/whisper/output_20s/external-data-youtube/video/20241103/'
TARGET_FILE_PATH = './target.txt'
AUDIO_CACHE_SUBDIR_NAME = ".audio_cache_sequential"
FRAMES_CACHE_SUBDIR_NAME = ".frames_cache"  # for video frames
VIDEO_EXTENSIONS = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm', '.mpeg', '.mpg']
MIN_AUDIO_DURATION_SEC = 0.1
MIN_AUDIO_SAMPLES = int(MIN_AUDIO_DURATION_SEC * 16000)
SEGMENT_DURATION = 20.0  # 20 seconds per segment
TARGET_FPS = 1  # 1 frames per second (1 frame every 1 seconds)

# --- Cached Model Loader ---
@functools.lru_cache(maxsize=None)
def get_whisper_model_sequential(model_name, device):
    print(f"[Sequential Mode, Device: {device}] Initializing Whisper model '{model_name}'...")
    model = whisper.load_model(model_name, device=device)
    print(f"[Sequential Mode, Device: {device}] Model '{model_name}' loaded successfully.")
    return model

# --- FFmpeg Audio Extraction ---
def extract_audio_ffmpeg_sequential(video_file_path, audio_file_path, pbar_instance, sample_rate=16000, channels=1):
    if os.path.exists(audio_file_path):
        return True, f"Audio already exists at {audio_file_path}"
    try:
        os.makedirs(os.path.dirname(audio_file_path), exist_ok=True)
        command = [
            'ffmpeg', '-nostdin', '-i', video_file_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate), '-ac', str(channels),
            '-y', audio_file_path
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        return True, f"Audio extracted successfully to {audio_file_path}"
    except subprocess.CalledProcessError as e:
        log_prefix = f"[File: {os.path.basename(video_file_path)}]"
        error_message = (
            f"{log_prefix} FFmpeg Error during audio extraction:\n"
            f"Stderr: {e.stderr.strip() if e.stderr else 'N/A'}\n"
            f"Stdout: {e.stdout.strip() if e.stdout else 'N/A'}"
        )
        if pbar_instance:
            pbar_instance.write(error_message)
        if os.path.exists(audio_file_path):
            try: os.remove(audio_file_path)
            except OSError: pass
        return False, error_message
    except FileNotFoundError:
        log_prefix = f"[File: {os.path.basename(video_file_path)}]"
        error_message = f"{log_prefix} CRITICAL: ffmpeg command not found."
        if pbar_instance:
            pbar_instance.write(error_message)
        return "ffmpeg_not_found", error_message

# --- Video Frame Extraction ---
def extract_frames_at_1fps(video_path: str, output_dir: str, pbar_instance) -> Tuple[bool, str, int]:
    """Extract frames from video at 1 FPS (one frame every 1 second)"""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log_prefix = f"[Video: {os.path.basename(video_path)}]"
            error_message = f"{log_prefix} Failed to open video file"
            if pbar_instance:
                pbar_instance.write(error_message)
            return False, error_message, 0
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Calculate frame interval for 1 FPS extraction
        frame_interval = int(fps / TARGET_FPS) if fps > TARGET_FPS else 1
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at 1 FPS intervals
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                frame_filename = f"frame_{extracted_count:06d}_t{timestamp:.2f}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        return True, f"Successfully extracted {extracted_count} frames", extracted_count
        
    except Exception as e:
        log_prefix = f"[Video: {os.path.basename(video_path)}]"
        error_message = f"{log_prefix} Error extracting frames: {str(e)}"
        if pbar_instance:
            pbar_instance.write(error_message)
        return False, error_message, 0

# --- Audio Segmentation ---
def split_audio_into_segments(audio_path: str, segment_duration: float = 20.0) -> List[Tuple[str, float, float]]:
    """split audio in segments of 20 seconds"""
    
    try:
        duration = librosa.get_duration(path=audio_path)
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return []
    
    segments = []
    current_time = 0.0
    segment_index = 0
    
    while current_time < duration:
        start_time = current_time
        end_time = min(current_time + segment_duration, duration)
        
        # splited audio segments path
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        segment_filename = f"{base_name}_segment_{segment_index:04d}.wav"
        segment_path = os.path.join(os.path.dirname(audio_path), segment_filename)
        
        segments.append((segment_path, start_time, end_time))
        current_time += segment_duration
        segment_index += 1
    
    return segments

def extract_audio_segment_ffmpeg(input_audio_path: str, output_path: str, start_time: float, end_time: float, pbar_instance=None) -> Tuple[bool, str]:
    """使用FFmpeg提取音频片段 - 静默模式"""
    
    if os.path.exists(output_path):
        return True, f"Segment already exists at {output_path}"
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        command = [
            'ffmpeg', '-nostdin', '-i', input_audio_path,
            '-ss', str(start_time), '-to', str(end_time),
            '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            '-y', output_path
        ]
        
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        return True, f"Audio segment extracted to {output_path}"
        
    except subprocess.CalledProcessError as e:
        log_prefix = f"[Segment: {start_time:.1f}s-{end_time:.1f}s]"
        error_message = f"{log_prefix} FFmpeg Error: {e.stderr.strip() if e.stderr else 'N/A'}"
        if pbar_instance:
            pbar_instance.write(error_message)
        if os.path.exists(output_path):
            try: os.remove(output_path)
            except OSError: pass
        return False, error_message

def transcribe_segments_separately(model, segments: List[Tuple[str, float, float]], pbar_instance, video_pbar=None) -> List[Dict]:
    """transcribe 20s audio segments with progress tracking"""
    import warnings
    import os
    
    # 临时禁用所有警告和设置环境变量
    warnings.filterwarnings("ignore")
    old_tqdm_disable = os.environ.get('TQDM_DISABLE', '')
    os.environ['TQDM_DISABLE'] = '1'
    
    all_segments = []
    
    # Create a sub-progress bar for transcription if video_pbar is provided
    transcription_pbar = tqdm(
        total=len(segments), 
        desc="Transcribing segments", 
        unit="seg", 
        leave=False,
        position=1 if video_pbar else 0,
        ncols=100  
    ) if video_pbar else None
    
    try:
        for segment_path, start_time, end_time in segments:
            log_prefix = f"[Segment: {start_time:.1f}s-{end_time:.1f}s]"
            
            try:
                # 转录单个片段 - 禁用Whisper内部进度条
                result = model.transcribe(
                    audio=segment_path,
                    language="en",
                    task="transcribe",
                    verbose=False,
                    # 禁用Whisper内部的进度显示
                    fp16=False
                )
                
                # 处理转录结果
                segment_text = result.get('text', '').strip()
                
                # 创建对齐的segment
                aligned_segment = {
                    'start': start_time,
                    'end': end_time,
                    'text': segment_text,
                    'frame_index': int(start_time * TARGET_FPS),  # 对应的帧索引
                    'segment_duration': end_time - start_time
                }
                
                all_segments.append(aligned_segment)
                
            except Exception as e:
                error_message = f"{log_prefix} Transcription failed: {str(e)}"
                if pbar_instance:
                    pbar_instance.write(error_message)
                # 添加空的segment以保持时间对齐
                all_segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': '[TRANSCRIPTION_FAILED]',
                    'frame_index': int(start_time * TARGET_FPS),
                    'segment_duration': end_time - start_time
                })
            
            # 更新转录进度
            if transcription_pbar:
                transcription_pbar.update(1)
                transcription_pbar.refresh()  # 强制刷新显示
    
    finally:
        # 恢复环境变量和警告设置
        os.environ['TQDM_DISABLE'] = old_tqdm_disable
        warnings.resetwarnings()
        
        if transcription_pbar:
            transcription_pbar.close()
    
    return all_segments

# --- Process Video with 20s Segments ---
def process_video_with_20s_segments(video_path, intermediate_audio_path, frames_output_dir, model, pbar):
    """processing with progress tracking"""
    
    # 1. 提取完整音频
    audio_extracted, audio_msg = extract_audio_ffmpeg_sequential(video_path, intermediate_audio_path, pbar)
    if not audio_extracted:
        raise Exception(f"Audio extraction failed: {audio_msg}")
    
    # 2. 提取1FPS帧
    frames_extracted, frames_msg, frame_count = extract_frames_at_1fps(video_path, frames_output_dir, pbar)
    if not frames_extracted:
        raise Exception(f"Frame extraction failed: {frames_msg}")
    
    # 3. 将音频分割成20秒片段
    audio_segments = split_audio_into_segments(intermediate_audio_path, SEGMENT_DURATION)
    
    if not audio_segments:
        raise Exception("Failed to create audio segments")
    
    # 4. 提取每个音频片段 - 静默处理
    successful_extractions = 0
    for segment_path, start_time, end_time in audio_segments:
        success, msg = extract_audio_segment_ffmpeg(intermediate_audio_path, segment_path, start_time, end_time, pbar)
        if success:
            successful_extractions += 1
    
    if successful_extractions != len(audio_segments):
        error_message = f"Audio segment extraction failed: {successful_extractions}/{len(audio_segments)} successful"
        if pbar:
            pbar.write(error_message)
    
    # 5. 分别转录每个片段 - 带进度显示
    aligned_segments = transcribe_segments_separately(model, audio_segments, pbar, video_pbar=pbar)
    
    # 6. 清理临时分段文件
    for segment_path, _, _ in audio_segments:
        try:
            if os.path.exists(segment_path):
                os.remove(segment_path)
        except OSError:
            pass
    
    return aligned_segments, frame_count

# --- Save Results ---
def save_aligned_results(output_path: str, aligned_segments: List[Dict], frame_info: Dict, pbar_instance=None):
    """Save time-aligned transcription and frame information"""
    results = {
        'segments': aligned_segments,
        'frame_info': frame_info,
        'metadata': {
            'segment_duration': SEGMENT_DURATION,
            'target_fps': TARGET_FPS
        }
    }
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        
        return True, f"JSON file saved successfully to {output_path}"
        
    except Exception as e:
        error_msg = f"Failed to save JSON file to {output_path}: {str(e)}"
        if pbar_instance:
            pbar_instance.write(f"✗ {error_msg}")
        return False, error_msg

def save_text_only(output_path: str, aligned_segments: List[Dict], pbar_instance=None):
    """Save text-only version with timestamps"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for seg in aligned_segments:
                f.write(f"[{seg['start']:.2f}-{seg['end']:.2f}s] {seg['text']}\n")
            f.flush()
            os.fsync(f.fileno())
        
        return True, f"Text file saved successfully to {output_path}"
        
    except Exception as e:
        error_msg = f"Failed to save text file to {output_path}: {str(e)}"
        if pbar_instance:
            pbar_instance.write(f"✗ {error_msg}")
        return False, error_msg

# --- Main Sequential Processing Logic ---
def main_sequential():
    import warnings
    import os
    
    # 在程序开始时设置环境变量，禁用额外的进度条
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    audio_cache_path_base = os.path.join(OUTPUT_DIR, AUDIO_CACHE_SUBDIR_NAME)
    frames_cache_path_base = os.path.join(OUTPUT_DIR, FRAMES_CACHE_SUBDIR_NAME)
    os.makedirs(audio_cache_path_base, exist_ok=True)
    os.makedirs(frames_cache_path_base, exist_ok=True)
    error_log_file_path = os.path.join(OUTPUT_DIR, "error_log_sequential.txt")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_str}")
    print(f"PyTorch version: {torch.__version__}")

    # Check dependencies
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        print("FFmpeg found.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("CRITICAL ERROR: FFmpeg not found or not executable. Please install FFmpeg and ensure it's in your system's PATH. Exiting.")
        return

    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        print("CRITICAL ERROR: OpenCV not found. Please install opencv-python. Exiting.")
        return

    try:
        import librosa
        print(f"Librosa version: {librosa.__version__}")
    except ImportError:
        print("CRITICAL ERROR: Librosa not found. Please install librosa. Exiting.")
        return

    try:
        model = get_whisper_model_sequential(MODEL_NAME, device_str)
    except Exception as e:
        print(f"Failed to load Whisper model: {e}. Exiting.")
        return

    # Read video file paths
    print(f"Reading video file paths from {TARGET_FILE_PATH}...")
    files_to_process_from_target = []
    if not os.path.exists(TARGET_FILE_PATH):
        print(f"CRITICAL ERROR: Target file {TARGET_FILE_PATH} not found. Exiting.")
        return
    
    try:
        with open(TARGET_FILE_PATH, "r", encoding="utf-8") as tf:
            for line in tf:
                video_path = line.strip()
                if video_path and os.path.isabs(video_path):
                    files_to_process_from_target.append(video_path)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not read {TARGET_FILE_PATH}: {e}. Exiting.")
        return

    if not files_to_process_from_target:
        print(f"No valid video paths found in {TARGET_FILE_PATH}.")
        return
    print(f"Read {len(files_to_process_from_target)} video paths from {TARGET_FILE_PATH}.")

    # Filter videos
    print("Filtering video files and checking for existing transcriptions...")
    files_to_process = []
    for video_path in files_to_process_from_target:
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}. Skipping.")
            continue

        try:
            common_prefix = os.path.commonpath([video_path, BASIC_DATA_PATH])
            if common_prefix != os.path.normpath(BASIC_DATA_PATH):
                print(f"Warning: Video path {video_path} is not within BASIC_DATA_PATH. Skipping.")
                continue
            relative_path_from_basic = os.path.relpath(video_path, BASIC_DATA_PATH)
        except ValueError:
            print(f"Warning: Could not determine relative path for {video_path}. Skipping.")
            continue

        base_rel_path_no_ext, _ = os.path.splitext(relative_path_from_basic)
        
        # Check for both JSON (aligned) and TXT (timestamped) outputs
        output_json_path = os.path.join(OUTPUT_DIR, base_rel_path_no_ext + "_aligned.json")
        output_txt_path = os.path.join(OUTPUT_DIR, base_rel_path_no_ext + "_timestamped.txt")

        if not (os.path.exists(output_json_path) and os.path.exists(output_txt_path)):
            files_to_process.append(video_path)

    if not files_to_process:
        print("No new video files found needing transcription.")
        return
    print(f"Found {len(files_to_process)} video files for transcription after filtering.")

    counts = {
        "success": 0, "skipped_existing": 0,
        "error_audio_extraction": 0, "error_frame_extraction": 0,
        "error_transcription": 0, "error_saving": 0, "error_general": 0
    }

    # 主进度条：总体处理进度
    with tqdm(total=len(files_to_process), desc="Processing Videos", unit="video", position=0, ncols=120, leave=True) as main_pbar:
        for video_path in files_to_process:
            video_name = os.path.basename(video_path)
            main_pbar.set_postfix_str(f"Current: {video_name}", refresh=True)
            status = "error_general"
            error_message = "Unknown processing error"

            try:
                relative_path_from_basic = os.path.relpath(video_path, BASIC_DATA_PATH)
                base_rel_path_no_ext, _ = os.path.splitext(relative_path_from_basic)
                
                # Output paths
                output_json_path = os.path.join(OUTPUT_DIR, base_rel_path_no_ext + "_aligned.json")
                output_txt_path = os.path.join(OUTPUT_DIR, base_rel_path_no_ext + "_timestamped.txt")
                intermediate_audio_path = os.path.join(audio_cache_path_base, base_rel_path_no_ext + ".wav")
                frames_output_dir = os.path.join(frames_cache_path_base, base_rel_path_no_ext)

                # Check if already processed
                if os.path.exists(output_json_path) and os.path.exists(output_txt_path):
                    status = "skipped_existing"
                    counts[status] += 1
                    main_pbar.update(1)
                    continue

                # Process video with 20s segments
                try:
                    aligned_segments, frame_count = process_video_with_20s_segments(
                        video_path, intermediate_audio_path, frames_output_dir, model, main_pbar
                    )
                    
                    # Prepare frame info
                    frame_info = {
                        'total_frames': frame_count,
                        'fps': TARGET_FPS,
                        'frame_dir': frames_output_dir
                    }
                    
                    # Save results
                    json_saved, json_msg = save_aligned_results(output_json_path, aligned_segments, frame_info, main_pbar)
                    if not json_saved:
                        status = "error_saving"
                        error_message = f"Failed to save JSON: {json_msg}"
                        main_pbar.write(error_message)
                        raise Exception("JSON saving failed")
                    
                    txt_saved, txt_msg = save_text_only(output_txt_path, aligned_segments, main_pbar)
                    if not txt_saved:
                        status = "error_saving"
                        error_message = f"Failed to save text: {txt_msg}"
                        main_pbar.write(error_message)
                        raise Exception("Text saving failed")
                    
                    # Verify files exist
                    if not os.path.exists(output_json_path):
                        status = "error_saving"
                        error_message = f"JSON file not found after saving: {output_json_path}"
                        main_pbar.write(error_message)
                        raise Exception("JSON file verification failed")
                    
                    if not os.path.exists(output_txt_path):
                        status = "error_saving"
                        error_message = f"Text file not found after saving: {output_txt_path}"
                        main_pbar.write(error_message)
                        raise Exception("Text file verification failed")
                    
                    status = "success"
                    
                except Exception as e:
                    if "saving" not in str(e).lower():
                        status = "error_transcription"
                        error_message = f"Processing error: {str(e)}"
                    main_pbar.write(f"[{video_name}] {error_message}")
                    raise

            except Exception as e:
                if status == "error_general":
                    error_message = f"Unhandled error: {str(e)}"
                main_pbar.write(f"[{video_name}] Error: {error_message}")
            
            counts[status] += 1
            if "error" in status:
                with open(error_log_file_path, "a", encoding="utf-8") as err_f:
                    err_f.write(f"{video_path}: {error_message}\n")
            
            main_pbar.update(1)

    # Print summary
    print("\n--- Processing Complete ---")
    for status_key, count_val in counts.items():
        if count_val > 0:
            print(f"{status_key.replace('_', ' ').capitalize()}: {count_val} files.")
    print(f"Total files processed: {len(files_to_process)}")

if __name__ == '__main__':
    main_sequential()
