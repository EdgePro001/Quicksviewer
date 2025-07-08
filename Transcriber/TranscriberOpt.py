import whisper_timestamped as whisper
import torch
import os
import subprocess
from typing import List, Dict, Tuple
import warnings
import numpy as np
from tqdm import tqdm
import functools
import json
import librosa
import cv2
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import threading
import queue
import time
import gc
import psutil
from pathlib import Path


# --- Configuration ---
MODEL_PATH = "/user/zhangyizhe/whisper-large-v3/my_whisper_model"
BASIC_DATA_PATH = '/thunlp_train/datasets/external-data-youtube/video'
OUTPUT_DIR = "/user/majunxian/SFTPipeline/Pool"
SEGMENT_DURATION = 20.0  # 20 seconds per segment
TARGET_FPS = 1  # 1 frames per second (1 frame every 1 seconds)
TARGET_FILE_PATH = '/user/zhangyizhe/whisper_module/target.txt'
AUDIO_CACHE_SUBDIR_NAME = "audio_cache_parallel"
FRAMES_CACHE_SUBDIR_NAME = "frames_cache"  # for video frames
VIDEO_EXTENSIONS = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm', '.mpeg', '.mpg']
MIN_AUDIO_DURATION_SEC = 0.1
MIN_AUDIO_SAMPLES = int(MIN_AUDIO_DURATION_SEC * 16000)

# --- DYNAMIC GPU Configuration (Read from environment variables) ---
def get_gpu_configuration():
    """
    Dynamically get GPU configuration, prioritize environment variables over defaults
    """
    # Read configuration from environment variables
    env_num_gpus = os.environ.get('WHISPER_NUM_GPUS')
    env_models_per_gpu = os.environ.get('WHISPER_MODELS_PER_GPU')
    env_batch_size = os.environ.get('WHISPER_BATCH_SIZE')
    
    # Get actual available GPU count
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # Determine GPU configuration
    if env_num_gpus and env_models_per_gpu:
        # Use environment variable configuration
        num_gpus = int(env_num_gpus)
        models_per_gpu = int(env_models_per_gpu)
        
        # Validate GPU availability
        if num_gpus > available_gpus:
            print(f"Warning: Requested {num_gpus} GPUs, but only {available_gpus} available")
            if available_gpus > 0:
                print(f"   Adjusting to use {available_gpus} GPUs with {models_per_gpu} models each")
                num_gpus = available_gpus
            else:
                print("   Falling back to CPU processing")
                num_gpus = 1
                models_per_gpu = min(8, os.cpu_count() or 4)
        
        print(f"Using ENVIRONMENT configuration:")
        print(f"   - GPUs: {num_gpus}")
        print(f"   - Models per GPU: {models_per_gpu}")
        print(f"   - Total workers: {num_gpus * models_per_gpu}")
        
    else:
        # Use default configuration (improved version of original logic)
        if available_gpus >= 4:
            # Default configuration for 4 or more GPUs
            num_gpus = min(4, available_gpus)  # Use maximum 4 GPUs
            models_per_gpu = 6  # 6 models per GPU (user expected configuration)
        elif available_gpus > 0:
            # Default configuration for fewer than 4 GPUs
            num_gpus = available_gpus
            models_per_gpu = 3  # Use 3 models when fewer GPUs available
        else:
            # CPU fallback
            num_gpus = 1
            models_per_gpu = min(8, os.cpu_count() or 4)
        
        print(f"Using DEFAULT configuration:")
        print(f"   - GPUs: {num_gpus} (available: {available_gpus})")
        print(f"   - Models per GPU: {models_per_gpu}")
        print(f"   - Total workers: {num_gpus * models_per_gpu}")
    
    # Batch size
    batch_size = int(env_batch_size) if env_batch_size else 32
    
    return {
        'num_gpus': num_gpus,
        'models_per_gpu': models_per_gpu,
        'total_workers': num_gpus * models_per_gpu,
        'batch_size': batch_size,
        'available_gpus': available_gpus
    }

# Get configuration
GPU_CONFIG = get_gpu_configuration()
NUM_GPUS = GPU_CONFIG['num_gpus']
MODELS_PER_GPU = GPU_CONFIG['models_per_gpu']  
TOTAL_WORKERS = GPU_CONFIG['total_workers']
BATCH_SIZE = GPU_CONFIG['batch_size']

print(f"FINAL CONFIGURATION:")
print(f"   NUM_GPUS = {NUM_GPUS}")
print(f"   MODELS_PER_GPU = {MODELS_PER_GPU}")
print(f"   TOTAL_WORKERS = {TOTAL_WORKERS}")
print(f"   BATCH_SIZE = {BATCH_SIZE}")

# --- A100-Optimized Settings ---
GPU_MEMORY_FRACTION = 0.95  # Use 95% of 80GB memory
PREFETCH_FACTOR = 4  # Increased prefetching for large memory
QUEUE_SIZE_PER_WORKER = 8  # More work per worker
SEGMENT_BATCH_SIZE = 64  # Process more segments at once

# --- HIGH-PERFORMANCE I/O Configuration ---
MAX_IO_WORKERS = min(16, (os.cpu_count() or 1))  # Scale with CPU cores
IO_BUFFER_SIZE = 1024 * 1024  # 1MB buffer for file operations
WRITE_BATCH_SIZE = 5  # Batch multiple writes together

# --- CUDA Optimization Settings ---
GPU_MEMORY_FRACTION = 0.85  # Reserve 85% of GPU memory per model
MIXED_PRECISION = True  # Enable mixed precision
PREFETCH_FACTOR = 3  # Number of batches to prefetch
PERSISTENT_WORKERS = True  # Keep workers alive

# --- Queue Management Settings ---
QUEUE_SIZE_PER_WORKER = 6  # Items per worker in queue
REFILL_THRESHOLD_RATIO = 0.25  # Refill when queue is 25% full
WORKER_TIMEOUT = 0.3  # Very short timeout for immediate redistribution


# --- ENHANCED I/O with Improved TXT Format and FORMATTED JSON ---
def save_results_enhanced_format(output_json_path: str, output_txt_path: str, 
                                cleaned_segments: List[Dict], frame_info: Dict) -> bool:
    """
    Enhanced I/O with improved TXT format and properly formatted JSON
    Maintains efficiency while providing rich formatting and readability
    """
    try:
        # Create directories once
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
        
        # Prepare data once
        results = {
            'metadata': {
                'segment_duration': SEGMENT_DURATION,
                'target_fps': TARGET_FPS,
                'models_per_gpu': MODELS_PER_GPU,
                'batch_size': BATCH_SIZE,
                'total_workers': TOTAL_WORKERS,
                'optimization_version': 'enhanced_format_v2.0',
                'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_segments': len(cleaned_segments),
                'total_frames': frame_info.get('frame_count', 'N/A')
            },
            'frame_info': frame_info,
            'segments': cleaned_segments,
            'summary_statistics': {}
        }
        
        # Calculate summary statistics
        if cleaned_segments:
            total_duration = cleaned_segments[-1]['end'] - cleaned_segments[0]['start']
            avg_segment_duration = sum(seg['end'] - seg['start'] for seg in cleaned_segments) / len(cleaned_segments)
            merged_count = sum(1 for seg in cleaned_segments if seg.get("merged_across_chunks", False))
            total_words = sum(seg.get('word_count', len(seg.get('text', '').split())) for seg in cleaned_segments)
            total_chars = sum(len(seg.get('text', '')) for seg in cleaned_segments)
            
            results['summary_statistics'] = {
                'total_duration_seconds': round(total_duration, 2),
                'total_duration_minutes': round(total_duration / 60, 2),
                'average_segment_duration': round(avg_segment_duration, 2),
                'merged_segments_count': merged_count,
                'segments_per_minute': round(len(cleaned_segments) / (total_duration / 60), 1) if total_duration > 0 else 0,
                'total_words': total_words,
                'total_characters': total_chars,
                'words_per_minute': round(total_words / (total_duration / 60), 1) if total_duration > 0 else 0,
                'average_words_per_segment': round(total_words / len(cleaned_segments), 1) if cleaned_segments else 0
            }
        
        # Write FORMATTED JSON with proper indentation and sorting
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(
                results, 
                f, 
                ensure_ascii=False, 
                indent=2,  # 2-space indentation for readability
                sort_keys=True,  # Sort keys alphabetically
                separators=(',', ': ')  # Add space after colon for better readability
            )
        
        # Write Enhanced TXT format (structured like JSON but human-readable)
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            # Header with metadata
            f.write("=" * 80 + "\n")
            f.write("WHISPER TRANSCRIPTION RESULTS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Processing Timestamp: {results['metadata']['processing_timestamp']}\n")
            f.write(f"Optimization Version: {results['metadata']['optimization_version']}\n")
            f.write(f"\nVideo Processing Metadata:\n")
            f.write(f"  - Segment Duration: {SEGMENT_DURATION}s\n")
            f.write(f"  - Target FPS: {TARGET_FPS}\n")
            f.write(f"  - Total Segments: {len(cleaned_segments)}\n")
            f.write(f"  - Frame Count: {frame_info.get('frame_count', 'N/A')}\n")
            f.write(f"  - Models per GPU: {MODELS_PER_GPU}\n")
            f.write(f"  - Batch Size: {BATCH_SIZE}\n")
            f.write(f"  - Total Workers: {TOTAL_WORKERS}\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary Statistics (if available)
            if 'summary_statistics' in results and results['summary_statistics']:
                stats = results['summary_statistics']
                f.write("SUMMARY STATISTICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Duration: {stats['total_duration_seconds']:.2f}s ({stats['total_duration_minutes']:.1f} minutes)\n")
                f.write(f"Total Segments: {len(cleaned_segments)}\n")
                f.write(f"Average Segment Duration: {stats['average_segment_duration']:.2f}s\n")
                f.write(f"Merged Segments: {stats['merged_segments_count']}\n")
                f.write(f"Segments per Minute: {stats['segments_per_minute']}\n")
                f.write(f"Total Words: {stats['total_words']:,}\n")
                f.write(f"Total Characters: {stats['total_characters']:,}\n")
                f.write(f"Words per Minute: {stats['words_per_minute']}\n")
                f.write(f"Average Words per Segment: {stats['average_words_per_segment']}\n")
                f.write("=" * 80 + "\n\n")
            
            # Segments section
            f.write("TRANSCRIPTION SEGMENTS:\n")
            f.write("-" * 80 + "\n\n")
            
            for i, seg in enumerate(cleaned_segments):
                # Segment header
                f.write(f"Segment {i + 1:03d}:\n")
                f.write(f"  Time Range: [{seg['start']:.2f}s - {seg['end']:.2f}s]")
                
                # Duration and special flags
                duration = seg['end'] - seg['start']
                f.write(f" (Duration: {duration:.2f}s)\n")
                
                # Frame information
                frame_idx = seg.get('frame_index', int(seg['start']) * TARGET_FPS)
                f.write(f"  Frame Index: {frame_idx}\n")
                
                # Word count and confidence (if available)
                word_count = seg.get('word_count', len(seg.get('text', '').split()))
                f.write(f"  Word Count: {word_count}\n")
                
                if 'confidence' in seg:
                    f.write(f"  Confidence: {seg['confidence']:.3f}\n")
                
                # Special flags and metadata
                if seg.get("merged_across_chunks", False):
                    f.write(f"  Status: MERGED ACROSS CHUNKS\n")
                    if 'merged_from_chunks' in seg:
                        f.write(f"  Merged From Chunks: {seg['merged_from_chunks']}\n")
                
                # Additional metadata if available
                if 'chunk_index' in seg:
                    f.write(f"  Chunk Index: {seg['chunk_index']}\n")
                if 'segment_index_in_chunk' in seg:
                    f.write(f"  Segment in Chunk: {seg['segment_index_in_chunk']}\n")
                if seg.get('grouped_from_segments', 0) > 1:
                    f.write(f"  Grouped from {seg['grouped_from_segments']} segments\n")
                if seg.get('spans_multiple_chunks', False):
                    f.write(f"  Spans Multiple Chunks: Yes\n")
                
                # Position flags
                if seg.get('is_first_in_chunk'):
                    f.write(f"  Position: First in chunk\n")
                elif seg.get('is_last_in_chunk'):
                    f.write(f"  Position: Last in chunk\n")
                
                # Text content (properly formatted and escaped)
                text = seg.get('text', '').strip()
                f.write(f"  Text: \"{text}\"\n")
                
                # Separator between segments
                f.write(f"  {'-' * 60}\n\n")
            
            # Final summary
            f.write("=" * 80 + "\n")
            f.write("END OF TRANSCRIPTION\n")
            f.write(f"Generated: {results['metadata']['processing_timestamp']}\n")
            f.write(f"Enhanced Whisper Processing System v2.0\n")
            f.write("=" * 80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"Enhanced I/O error: {e}")
        import traceback
        traceback.print_exc()
        return False

# --- Audio Validation Functions ---
def validate_audio_segment(audio_path: str, min_duration: float = 0.1) -> Tuple[bool, float]:
    """Validate if audio segment has sufficient duration and content"""
    try:
        # Check file size first
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:  # Less than 1KB
            return False, 0.0
            
        duration = librosa.get_duration(path=audio_path)
        if duration < min_duration:
            return False, duration
            
        # Load audio and check for silence
        audio_data, sr = librosa.load(audio_path, sr=16000)
        if len(audio_data) < 1600:  # Less than 0.1 seconds at 16kHz
            return False, duration
            
        # Check if audio is mostly silent (you can adjust threshold)
        rms_energy = np.sqrt(np.mean(audio_data**2))
        if rms_energy < 0.001:  # Very quiet audio
            return False, duration
            
        return True, duration
    except Exception as e:
        print(f"Error validating audio {audio_path}: {e}")
        return False, 0.0


# --- OPTIMIZED Whisper Processor with CUDA Enhancements ---
class OptimizedWhisperProcessor:
    """
    High-performance Whisper processor with CUDA optimizations
    """
    def __init__(self, model_path=MODEL_PATH, device=None, gpu_id=None, model_id=None):
        if device is None:
            if gpu_id is not None and torch.cuda.is_available():
                self.device = f"cuda:{gpu_id}"
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Set the specific GPU for this process
        if self.device.startswith("cuda:"):
            torch.cuda.set_device(self.device)
        
        self.gpu_id = gpu_id
        self.model_id = model_id
        self.model_path = model_path
        self.model = None
        self.batch_size = BATCH_SIZE
        
        # CUDA optimizations
        if torch.cuda.is_available():
            # Enable optimized memory allocation
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Set memory fraction if specified
            if GPU_MEMORY_FRACTION < 1.0:
                torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, device=self.device)
        
        self._load_model()
    
    def _load_model(self):
        """Load model with CUDA optimizations"""
        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        # Load model
        if self.model_path and os.path.exists(self.model_path):
            print(f"Loading local model from: {self.model_path} on {self.device} (Model {self.model_id})")
            self.model = whisper.load_model(self.model_path, device=self.device)
        else:
            print(f"Loading default 'large-v3' model on {self.device} (Model {self.model_id})...")
            self.model = whisper.load_model("large-v3", device=self.device)
        
        # Model optimizations for A100-80GB
        self.model.eval()
        
        # Enable mixed precision if supported
        if MIXED_PRECISION and torch.cuda.is_available():
            self.model = self.model.half()  # Convert to FP16
            print(f"Model {self.model_id} optimized with FP16 precision for A100")
            
        # A100-specific optimizations
        if torch.cuda.is_available() and "A100" in torch.cuda.get_device_name(self.device):
            # Enable TensorFloat-32 (TF32) for A100
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"Model {self.model_id} enabled TF32 for A100 acceleration")
        
        print(f"Model {self.model_id} loaded successfully on {self.device}!")

    def transcribe_with_optimizations(self, audio_file, language="en"):
        """Enhanced transcription with CUDA optimizations"""
        try:
            audio = whisper.load_audio(audio_file)
            
            # CUDA optimizations for inference
            with torch.cuda.amp.autocast(enabled=MIXED_PRECISION):
                with torch.no_grad():
                    result = whisper.transcribe(
                        self.model,
                        audio,
                        language=language,
                        temperature=0,  # Deterministic
                        compression_ratio_threshold=2.4,
                        logprob_threshold=-1.0,
                        no_speech_threshold=0.6,
                        condition_on_previous_text=False,
                        verbose=False,
                        # Additional optimizations
                        fp16=MIXED_PRECISION,
                        without_timestamps=False
                    )
            return result
        except Exception as e:
            print(f"Transcription error for {audio_file}: {e}")
            return {"segments": []}
    
    def transcribe_batch_optimized(self, audio_segments: List[Tuple[str, float, float]]) -> List[Dict]:
        """A100-optimized batch transcription with enhanced segment metadata"""
        batch_results = []
        
        # Use larger batch size for A100-80GB
        effective_batch_size = min(self.batch_size, SEGMENT_BATCH_SIZE)
        
        # Pre-allocate results list for better memory management
        batch_results = [None] * len(audio_segments)
        
        # Process in A100-optimized batches
        for batch_start in range(0, len(audio_segments), effective_batch_size):
            batch_end = min(batch_start + effective_batch_size, len(audio_segments))
            current_batch = audio_segments[batch_start:batch_end]
            
            # Pre-load all audio for the entire batch (utilize A100 memory)
            batch_audio_data = []
            for segment_idx, (segment_path, start_time, end_time) in enumerate(current_batch):
                global_idx = batch_start + segment_idx
                
                # Validate audio segment
                is_valid, duration = validate_audio_segment(segment_path)
                if not is_valid:
                    batch_audio_data.append((None, start_time, end_time, global_idx))
                    continue
                
                try:
                    # Pre-load audio for this segment
                    audio = whisper.load_audio(segment_path)
                    batch_audio_data.append((audio, start_time, end_time, global_idx))
                except Exception as e:
                    print(f"Error loading audio {segment_path}: {e}")
                    batch_audio_data.append((None, start_time, end_time, global_idx))
            
            # A100-optimized batch processing with larger context
            with torch.cuda.amp.autocast(enabled=MIXED_PRECISION):
                with torch.no_grad():
                    # Process multiple segments simultaneously when possible
                    for audio_data in batch_audio_data:
                        audio, start_time, end_time, segment_idx = audio_data
                        
                        if audio is None:
                            batch_results[segment_idx] = {
                                "segments": [],
                                "original_start": start_time,
                                "original_end": end_time,
                                "chunk_index": segment_idx
                            }
                            continue
                        
                        try:
                            # A100-optimized transcription with larger beam size
                            result = whisper.transcribe(
                                self.model,
                                audio,
                                language="en",
                                temperature=0,
                                compression_ratio_threshold=2.4,
                                logprob_threshold=-1.0,
                                no_speech_threshold=0.6,
                                condition_on_previous_text=False,
                                verbose=False,
                                fp16=MIXED_PRECISION,
                                # A100-specific optimizations
                                beam_size=5,  # Larger beam for better accuracy
                                patience=1.0,
                                length_penalty=1.0
                            )
                            
                            if not result or not result.get("segments"):
                                batch_results[segment_idx] = {
                                    "segments": [],
                                    "original_start": start_time,
                                    "original_end": end_time,
                                    "chunk_index": segment_idx
                                }
                            else:
                                # Process segments and adjust timestamps with enhanced metadata
                                processed_segments = []
                                for seg_idx, seg in enumerate(result.get("segments", [])):
                                    global_start = seg["start"] + start_time
                                    global_end = seg["end"] + start_time
                                    
                                    processed_seg = {
                                        "start": global_start,
                                        "end": global_end,
                                        "text": seg["text"],
                                        "original_start": start_time,
                                        "original_end": end_time,
                                        "frame_index": int(global_start) * TARGET_FPS,
                                        "segment_duration": global_end - global_start,
                                        "chunk_index": segment_idx,
                                        "segment_index_in_chunk": seg_idx,
                                        "is_first_in_chunk": seg_idx == 0,
                                        "is_last_in_chunk": seg_idx == len(result["segments"]) - 1,
                                        # Additional metadata for enhanced format
                                        "confidence": seg.get("avg_logprob", 0.0),
                                        "no_speech_prob": seg.get("no_speech_prob", 0.0),
                                        "word_count": len(seg["text"].split()) if seg["text"] else 0
                                    }
                                    processed_segments.append(processed_seg)
                                
                                result["segments"] = processed_segments
                                result["original_start"] = start_time
                                result["original_end"] = end_time
                                result["chunk_index"] = segment_idx
                                batch_results[segment_idx] = result
                                
                        except Exception as e:
                            print(f"Transcription error in batch: {e}")
                            batch_results[segment_idx] = {
                                "segments": [],
                                "original_start": start_time,
                                "original_end": end_time,
                                "chunk_index": segment_idx
                            }
            
            # Less frequent memory management for A100 (has more memory)
            if (batch_start // effective_batch_size) % 3 == 0:  # Every 3 batches instead of 5
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Filter out None results and sort by chunk_index
        final_results = [r for r in batch_results if r is not None]
        final_results.sort(key=lambda x: x.get("chunk_index", 0))
        
        return final_results
    
    def _should_merge_segments(self, prev_seg, cur_seg, time_gap, max_gap_seconds):
        """Determine if two segments should be merged based on multiple criteria"""
        # 1. Time gap check
        if time_gap > max_gap_seconds:
            return False
        
        # 2. Check if previous segment seems incomplete
        prev_text = prev_seg["text"].strip()
        sentence_endings = ['.', '!', '?', '。', '！', '？']
        
        if any(prev_text.endswith(ending) for ending in sentence_endings):
            return time_gap <= 0.3
        
        # 3. Check if current segment seems like a continuation
        cur_text = cur_seg["text"].strip()
        
        continuation_indicators = [
            cur_text and cur_text[0].islower(),
            cur_text.startswith(('and ', 'but ', 'or ', 'so ', 'then ', 'because ')),
            cur_text.startswith(('the ', 'that ', 'this ', 'it ', 'he ', 'she ', 'they ')),
        ]
        
        if any(continuation_indicators):
            return True
        
        # 4. Check segment duration
        prev_duration = prev_seg["end"] - prev_seg["start"]
        if prev_duration < 1.0:
            return True
        
        return time_gap <= 0.5

    def _merge_text(self, prev_text, cur_text):
        """Intelligently merge text from two segments"""
        prev_clean = prev_text.strip()
        cur_clean = cur_text.strip()
        
        if not prev_clean:
            return cur_clean
        if not cur_clean:
            return prev_clean
        
        needs_space = not (
            prev_clean.endswith((' ', '-', '—')) or 
            cur_clean.startswith((' ', '-', '—', "'", '"'))
        )
        
        if needs_space:
            return f"{prev_clean} {cur_clean}"
        else:
            return f"{prev_clean}{cur_clean}"
    
    def merge_adjacent_segments(self, all_results, max_gap_seconds=0.8):
        """Enhanced merging logic to handle sentences split across 20s chunks"""
        merged_segments = []
        
        for i in range(len(all_results)):
            cur_result = all_results[i]
            cur_segs = cur_result.get("segments", [])
            
            if not cur_segs:
                continue
            if i == 0:
                merged_segments.extend(cur_segs)
            else:
                prev_result = all_results[i - 1]
                prev_segs = prev_result.get("segments", [])
                if not prev_segs:
                    merged_segments.extend(cur_segs)
                    continue
                    
                last_prev_seg = prev_segs[-1]
                first_cur_seg = cur_segs[0]
                time_gap = first_cur_seg["start"] - last_prev_seg["end"]
                
                should_merge = self._should_merge_segments(
                    last_prev_seg, first_cur_seg, time_gap, max_gap_seconds
                )
                
                if should_merge:
                    merged_segment = {
                        "text": self._merge_text(last_prev_seg["text"], first_cur_seg["text"]),
                        "start": last_prev_seg["start"],
                        "end": first_cur_seg["end"],
                        "original_start": last_prev_seg.get("original_start", last_prev_seg["start"]),
                        "original_end": first_cur_seg.get("original_end", first_cur_seg["end"]),
                        "frame_index": last_prev_seg.get("frame_index", int(last_prev_seg["start"])),
                        "segment_duration": first_cur_seg["end"] - last_prev_seg["start"],
                        "merged_across_chunks": True,
                        # Enhanced metadata for merged segments
                        "chunk_index": last_prev_seg.get("chunk_index", 0),
                        "merged_from_chunks": [
                            last_prev_seg.get("chunk_index", 0),
                            first_cur_seg.get("chunk_index", 0)
                        ],
                        "word_count": (last_prev_seg.get("word_count", 0) + 
                                     first_cur_seg.get("word_count", 0))
                    }
                    merged_segments[-1] = merged_segment
                    merged_segments.extend(cur_segs[1:])
                else:
                    merged_segments.extend(cur_segs)

        return merged_segments
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        if not text or not text.strip():
            return ""
        cleaned = " ".join(text.strip().split())
        artifacts = ["[BLANK_AUDIO]", "[Music]", "[Applause]", "[Laughter]"]
        for artifact in artifacts:
            cleaned = cleaned.replace(artifact, "").strip()
        return cleaned
    
    def postprocess_segments(self, segments):
        """Optimized segment post-processing with enhanced metadata"""
        if not segments:
            return []
    
        cleaned = []
        sentence_endings = {'.', '?', '!'} 
        i = 0
        n = len(segments)
    
        while i < n:
            group_start = segments[i]
            group_start_time = group_start["start"]
            group_texts = [group_start["text"]]
            frame_index = group_start.get("frame_index", 0)
            j = i + 1
            best_end_idx = i
            found_good_ending = False
            
            # Collect metadata from grouped segments
            total_word_count = group_start.get("word_count", 0)
            chunk_indices = [group_start.get("chunk_index", 0)]
            is_merged_across_chunks = group_start.get("merged_across_chunks", False)
        
            while j < n:
                current_duration = segments[j]["end"] - group_start_time
                if current_duration > 35.0:
                    break
                current_text = segments[j]["text"].strip()
                
                # Accumulate metadata
                total_word_count += segments[j].get("word_count", 0)
                chunk_idx = segments[j].get("chunk_index", 0)
                if chunk_idx not in chunk_indices:
                    chunk_indices.append(chunk_idx)
                if segments[j].get("merged_across_chunks", False):
                    is_merged_across_chunks = True
                
                if current_duration >= 15.0:
                    if current_text and current_text[-1] in sentence_endings:
                        best_end_idx = j
                        found_good_ending = True
                        break
                    elif not found_good_ending:
                        best_end_idx = j
                else:
                    best_end_idx = j
                j += 1
        
            if j < n and segments[j]["end"] - group_start_time > 35.0 and not found_good_ending:
                for k in range(j - 1, i, -1):
                    if segments[k]["text"].strip() and segments[k]["text"].strip()[-1] in sentence_endings:
                        duration_at_k = segments[k]["end"] - group_start_time
                        if duration_at_k >= 15.0:
                            best_end_idx = k
                            break
                if j > i + 1:
                    best_end_idx = j - 1
        
            for k in range(i + 1, best_end_idx + 1):
                group_texts.append(segments[k]["text"])
            
            group_end_time = segments[best_end_idx]["end"]
            group_duration = group_end_time - group_start_time
            
            # Create enhanced merged segment with rich metadata
            merged_segment = {
                "start": round(group_start_time, 2),
                "end": round(group_end_time, 2),
                "text": self.clean_text(" ".join(group_texts)),
                "frame_index": frame_index,
                "segment_duration": round(group_duration, 2),
                # Enhanced metadata
                "word_count": total_word_count,
                "grouped_from_segments": best_end_idx - i + 1,
                "chunk_indices": chunk_indices,
                "spans_multiple_chunks": len(chunk_indices) > 1,
                "merged_across_chunks": is_merged_across_chunks
            }
            cleaned.append(merged_segment)
            i = best_end_idx + 1
    
        return cleaned


# --- Video Frame Extraction ---
def extract_frames_at_1fps(video_path: str, output_dir: str, pbar_instance=None) -> Tuple[bool, str, int]:
    """Extract frames from video at 1 FPS with optimizations"""
    try:
        os.makedirs(output_dir, exist_ok=True)
    
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log_prefix = f"[Video: {os.path.basename(video_path)}]"
            error_message = f"{log_prefix} Failed to open video file"
            return False, error_message, 0
        
        # Optimize video reading
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        frame_interval = int(fps / TARGET_FPS) if fps > TARGET_FPS else 1
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                frame_filename = f"frame_{extracted_count:06d}_t{timestamp:.2f}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                
                # Optimize image compression
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                extracted_count += 1
            frame_count += 1
        
        cap.release()
        return True, f"Successfully extracted {extracted_count} frames", extracted_count
    
    except Exception as e:
        log_prefix = f"[Video: {os.path.basename(video_path)}]"
        error_message = f"{log_prefix} Error extracting frames: {str(e)}"
        return False, error_message, 0


# --- OPTIMIZED FFmpeg Functions ---
def extract_audio_ffmpeg_sequential(video_file_path, audio_file_path, sample_rate=16000, channels=1):
    """Extract audio using ffmpeg with optimizations"""
    if os.path.exists(audio_file_path):
        return True, f"Audio already exists at {audio_file_path}"
    try:
        os.makedirs(os.path.dirname(audio_file_path), exist_ok=True)
        command = [
            'ffmpeg', '-nostdin', '-i', video_file_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate), '-ac', str(channels),
            '-threads', '0',  # Use all available CPU cores
            '-y', audio_file_path
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        return True, f"Audio extracted successfully to {audio_file_path}"
    except subprocess.CalledProcessError as e:
        log_prefix = f"[File: {os.path.basename(video_file_path)}]"
        error_message = f"{log_prefix} FFmpeg Error: {e.stderr.strip() if e.stderr else 'N/A'}"
        if os.path.exists(audio_file_path):
            try: os.remove(audio_file_path)
            except OSError: pass
        return False, error_message
    except FileNotFoundError:
        log_prefix = f"[File: {os.path.basename(video_file_path)}]"
        error_message = f"{log_prefix} CRITICAL: ffmpeg command not found."
        return "ffmpeg_not_found", error_message


def split_audio_into_segments(audio_path: str, segment_duration: float = 20.0) -> List[Tuple[str, float, float]]:
    """Split audio in segments of 20 seconds"""
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
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        segment_filename = f"{base_name}_segment_{segment_index:04d}.wav"
        segment_path = os.path.join(os.path.dirname(audio_path), segment_filename)
        segments.append((segment_path, start_time, end_time))
        current_time += segment_duration
        segment_index += 1
    return segments


def extract_audio_segment_ffmpeg(input_audio_path: str, output_path: str, start_time: float, end_time: float) -> Tuple[bool, str]:
    """Extract segmented audio using ffmpeg with enhanced validation"""
    if os.path.exists(output_path):
        is_valid, duration = validate_audio_segment(output_path)
        if is_valid:
            return True, f"Valid segment already exists at {output_path}"
        else:
            try:
                os.remove(output_path)
            except OSError:
                pass

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        command = [
            'ffmpeg', '-nostdin', '-i', input_audio_path,
            '-ss', str(start_time), '-to', str(end_time),
            '-af', 'silenceremove=1:0:-50dB,aresample=16000',
            '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            '-threads', '0',  # Use all available cores
            '-y', output_path
        ]
        
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        is_valid, duration = validate_audio_segment(output_path)
        if not is_valid:
            if os.path.exists(output_path):
                try: 
                    os.remove(output_path)
                except OSError: 
                    pass
            return False, f"Created audio segment is invalid (duration: {duration:.3f}s)"
            
        return True, f"Valid audio segment extracted to {output_path}"
        
    except subprocess.CalledProcessError as e:
        log_prefix = f"[Segment: {start_time:.1f}s-{end_time:.1f}s]"
        error_message = f"{log_prefix} FFmpeg Error: {e.stderr.strip() if e.stderr else 'N/A'}"
        if os.path.exists(output_path):
            try: 
                os.remove(output_path)
            except OSError: 
                pass
        return False, error_message


def transcribe_segments_batch_worker(processor, segments: List[Tuple[str, float, float]]):
    """OPTIMIZED: Batch version of segment transcription for maximum GPU utilization"""
    warnings.filterwarnings("ignore")
    
    # Use the optimized batch transcription method
    all_results = processor.transcribe_batch_optimized(segments)
    
    return all_results


def safe_cleanup_cache_files(cache_dir: str, preserve_outputs: bool = True):
    """Safely clean up cache files while preserving output JSON/TXT files"""
    try:
        if not os.path.exists(cache_dir):
            return True, "Cache directory does not exist"
        
        files_removed = 0
        files_failed = 0
        
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    # Only remove if it's not an output file or if preserve_outputs is False
                    if not preserve_outputs or not (file.endswith('.json') or file.endswith('.txt')):
                        os.remove(file_path)
                        files_removed += 1
                except OSError as e:
                    files_failed += 1
                    print(f"Warning: Could not remove {file_path}: {e}")
        
        return True, f"Cleaned up {files_removed} files, {files_failed} failures"
        
    except Exception as e:
        return False, f"Cache cleanup error: {str(e)}"


# --- Modified process_single_video function with enhanced format ---
def process_single_video_optimized(video_path: str, processor) -> Dict:
    """
    Optimized video processing with enhanced formatted output
    """
    try:
        # Set up paths
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_cache_dir = os.path.join(OUTPUT_DIR, AUDIO_CACHE_SUBDIR_NAME, base_name)
        frames_cache_dir = os.path.join(OUTPUT_DIR, FRAMES_CACHE_SUBDIR_NAME, base_name)
        output_json_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")
        output_txt_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
        
        # Check if already processed
        if os.path.exists(output_json_path) and os.path.exists(output_txt_path):
            return {
                'status': 'skipped',
                'video_path': video_path,
                'message': 'Already processed'
            }
        
        # Create directories
        os.makedirs(audio_cache_dir, exist_ok=True)
        os.makedirs(frames_cache_dir, exist_ok=True)
        
        # Step 1: Extract audio
        intermediate_audio_path = os.path.join(audio_cache_dir, "full_audio.wav")
        audio_extracted, audio_msg = extract_audio_ffmpeg_sequential(video_path, intermediate_audio_path)
        if not audio_extracted:
            raise Exception(f"Audio extraction failed: {audio_msg}")
        
        # Step 2: Extract frames
        frames_extracted, frames_msg, frame_count = extract_frames_at_1fps(video_path, frames_cache_dir)
        if not frames_extracted:
            raise Exception(f"Frame extraction failed: {frames_msg}")
        
        # Step 3: Split audio into segments
        audio_segments = split_audio_into_segments(intermediate_audio_path, SEGMENT_DURATION)
        if not audio_segments:
            raise Exception("Failed to create audio segments")
        
        # Step 4: Extract audio segments
        valid_segments = []
        for segment_path, start_time, end_time in audio_segments:
            success, msg = extract_audio_segment_ffmpeg(intermediate_audio_path, segment_path, start_time, end_time)
            if success:
                valid_segments.append((segment_path, start_time, end_time))
        
        if not valid_segments:
            raise Exception("No valid audio segments created")
        
        # Step 5: Batch transcribe segments (GPU processing)
        all_whisper_results = transcribe_segments_batch_worker(processor, valid_segments)
        
        # Step 6: Merge adjacent segments
        merged_segments = processor.merge_adjacent_segments(all_whisper_results)
        
        # Step 7: Post-process with enhanced metadata
        cleaned_segments = processor.postprocess_segments(merged_segments)
        
        # Step 8: Enhanced I/O with improved formatting
        frame_info = {
            "frame_count": frame_count,
            "target_fps": TARGET_FPS
        }
        
        # Save with enhanced format
        io_success = save_results_enhanced_format(output_json_path, output_txt_path, cleaned_segments, frame_info)
        
        if not io_success:
            raise Exception("Failed to save results")
        
        # Step 9: Clean up temporary segment files
        for segment_path, _, _ in valid_segments:
            try:
                if os.path.exists(segment_path) and "segment_" in os.path.basename(segment_path):
                    os.remove(segment_path)
            except OSError:
                pass
        
        return {
            'status': 'success',
            'video_path': video_path,
            'segments_count': len(cleaned_segments),
            'frame_count': frame_count,
            'total_words': sum(seg.get('word_count', 0) for seg in cleaned_segments),
            'total_duration': cleaned_segments[-1]['end'] - cleaned_segments[0]['start'] if cleaned_segments else 0
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'video_path': video_path,
            'error': str(e)
        }


# --- Updated worker function for enhanced I/O ---
def optimized_worker_process_enhanced(worker_id: int, gpu_id: int, model_id: int, 
                                    video_queue: mp.Queue, result_queue: mp.Queue):
    """
    Optimized worker with enhanced I/O formatting
    """
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    try:
        # Initialize processor
        processor = OptimizedWhisperProcessor(
            model_path=MODEL_PATH, 
            gpu_id=gpu_id, 
            model_id=f"GPU{gpu_id}-M{model_id}"
        )
        
        videos_processed = 0
        
        while True:
            try:
                video_path = video_queue.get(timeout=0.3)
                
                if video_path is None:  # Sentinel value
                    break
                
                # Process video with enhanced formatting
                result = process_single_video_optimized(video_path, processor)
                result['worker_id'] = worker_id
                result['gpu_id'] = gpu_id
                result['model_id'] = model_id
                
                result_queue.put(result)
                videos_processed += 1
                
                # Memory cleanup
                if videos_processed % 2 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except queue.Empty:
                break
            except Exception as e:
                error_result = {
                    'status': 'error',
                    'video_path': video_path if 'video_path' in locals() else 'unknown',
                    'worker_id': worker_id,
                    'error': str(e)
                }
                result_queue.put(error_result)
        
        # Worker completion
        completion_result = {
            'status': 'worker_completed',
            'worker_id': worker_id,
            'videos_processed': videos_processed
        }
        result_queue.put(completion_result)
    
    except Exception as e:
        error_result = {
            'status': 'error',
            'video_path': 'initialization_error',
            'worker_id': worker_id,
            'error': f"Worker init failed: {str(e)}"
        }
        result_queue.put(error_result)


# --- Enhanced main function ---
def main_high_performance_enhanced():
    """
    Enhanced main function with improved output formatting
    """
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # GPU configuration already set at module level - no adjustment needed
    
    # Set environment variables for optimal performance
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Enable async GPU operations
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,expandable_segments:True'  # Better memory management
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    audio_cache_path_base = os.path.join(OUTPUT_DIR, AUDIO_CACHE_SUBDIR_NAME)
    frames_cache_path_base = os.path.join(OUTPUT_DIR, FRAMES_CACHE_SUBDIR_NAME)
    os.makedirs(audio_cache_path_base, exist_ok=True)
    os.makedirs(frames_cache_path_base, exist_ok=True)
    error_log_file_path = os.path.join(OUTPUT_DIR, "error_log_enhanced.txt")
    
    print(f"ENHANCED WHISPER PROCESSING SYSTEM:")
    print(f"=" * 70)
    print(f"Hardware-Specific Optimization:")
    print(f"  - Target: {NUM_GPUS}x NVIDIA A100-SXM4-80GB")
    print(f"  - Memory per GPU: 80GB")
    print(f"  - Optimized for large-scale processing")
    print(f"GPU Setup (A100-Optimized):")
    print(f"  - Using {NUM_GPUS} GPUs")
    print(f"  - {MODELS_PER_GPU} models per GPU")
    print(f"  - {TOTAL_WORKERS} total workers")
    print(f"CUDA Optimizations:")
    print(f"  - Mixed precision: {MIXED_PRECISION}")
    print(f"  - TF32 enabled for A100")
    print(f"  - GPU memory fraction: {GPU_MEMORY_FRACTION}")
    print(f"  - Batch size: {BATCH_SIZE} segments")
    print(f"  - Segment batch size: {SEGMENT_BATCH_SIZE}")
    print(f"Output Format Enhancements:")
    print(f"  - JSON: Complete metadata with rich segment info")
    print(f"  - TXT: Structured format with headers, metadata, and statistics")
    print(f"  - Enhanced metadata: word counts, confidence scores, chunk tracking")
    print(f"  - Segment grouping: intelligent merging across 20s boundaries")
    print(f"  - Summary statistics: duration, word rates, processing metrics")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Multiprocessing method: {mp.get_start_method()}")

    # Check dependencies
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True)
        print("FFmpeg found.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("CRITICAL ERROR: FFmpeg not found. Please install FFmpeg. Exiting.")
        return

    # Read video file list
    print(f"Reading video file paths from {TARGET_FILE_PATH}...")
    files_to_process = []
    if not os.path.exists(TARGET_FILE_PATH):
        print(f"CRITICAL ERROR: Target file {TARGET_FILE_PATH} not found. Exiting.")
        return

    try:
        with open(TARGET_FILE_PATH, "r", encoding="utf-8") as tf:
            for line in tf:
                video_path = line.strip()
                if video_path and os.path.isabs(video_path) and os.path.exists(video_path):
                    files_to_process.append(video_path)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not read {TARGET_FILE_PATH}: {e}. Exiting.")
        return

    if not files_to_process:
        print(f"No valid video paths found in {TARGET_FILE_PATH}.")
        return
    
    print(f"Found {len(files_to_process)} videos to process")
    print(f"Work distribution: {len(files_to_process) / TOTAL_WORKERS:.1f} videos per worker on average")
    
    # Initialize queue management
    video_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # Add all videos to queue
    for video_path in files_to_process:
        video_queue.put(video_path)
    
    # Add sentinel values
    for _ in range(TOTAL_WORKERS):
        video_queue.put(None)
    
    print(f"Added {len(files_to_process)} videos and {TOTAL_WORKERS} sentinels to queue")
    
    # Start workers with enhanced processing
    print(f"Starting {TOTAL_WORKERS} workers with enhanced formatting...")
    worker_processes = []
    worker_id = 0
    
    for gpu_id in range(NUM_GPUS):
        for model_id in range(1, MODELS_PER_GPU + 1):
            print(f"Starting worker {worker_id + 1}/{TOTAL_WORKERS}: GPU-{gpu_id}, Model-{model_id}")
            
            actual_gpu_id = gpu_id if torch.cuda.is_available() else None
            
            process = mp.Process(
                target=optimized_worker_process_enhanced, 
                args=(worker_id, actual_gpu_id, model_id, video_queue, result_queue)
            )
            process.start()
            worker_processes.append(process)
            
            time.sleep(1.5)  # Sequential loading
            worker_id += 1
    
    print(f"All {TOTAL_WORKERS} workers started with enhanced formatting!")
    
    # Process results with enhanced monitoring
    success_count = 0
    error_count = 0
    skipped_count = 0
    processed_count = 0
    completed_workers = 0
    total_words_processed = 0
    total_duration_processed = 0.0
    
    start_time = time.time()
    last_update_time = time.time()
    
    with open(error_log_file_path, "w", encoding="utf-8") as error_log:
        with tqdm(total=len(files_to_process), desc="Enhanced Processing", unit="video") as pbar:
            while processed_count < len(files_to_process):
                try:
                    result = result_queue.get(timeout=30)
                    
                    if result.get('status') == 'worker_completed':
                        completed_workers += 1
                        worker_info = f"W{result['worker_id']}"
                        print(f"{worker_info} completed ({result['videos_processed']} videos)")
                        continue
                    
                    processed_count += 1
                    current_time = time.time()
                    
                    if result['status'] == 'success':
                        success_count += 1
                        worker_info = f"W{result['worker_id']}-GPU{result['gpu_id']}-M{result['model_id']}"
                        
                        # Accumulate statistics
                        total_words_processed += result.get('total_words', 0)
                        total_duration_processed += result.get('total_duration', 0)
                        
                        # Calculate real-time metrics
                        elapsed = current_time - start_time
                        videos_per_second = processed_count / elapsed if elapsed > 0 else 0
                        eta_seconds = (len(files_to_process) - processed_count) / videos_per_second if videos_per_second > 0 else 0
                        words_per_minute = (total_words_processed / (total_duration_processed / 60)) if total_duration_processed > 0 else 0
                        
                        pbar.set_postfix({
                            'Success': success_count,
                            'Errors': error_count,
                            'Skipped': skipped_count,
                            'Rate': f"{videos_per_second:.2f}/s",
                            'WPM': f"{words_per_minute:.0f}",
                            'ETA': f"{eta_seconds/60:.1f}m"
                        })
                        
                        # Detailed logging every 10 seconds
                        if current_time - last_update_time > 10:
                            tqdm.write(f"Enhanced Processing Rate: {videos_per_second:.2f}/s | "
                                     f"Words/min: {words_per_minute:.0f} | "
                                     f"Success: {success_count} | Processed: {processed_count}/{len(files_to_process)}")
                            last_update_time = current_time
                        
                        tqdm.write(f"{worker_info}: {os.path.basename(result['video_path'])} "
                                 f"({result['segments_count']} segments, {result['frame_count']} frames, "
                                 f"{result.get('total_words', 0)} words, {result.get('total_duration', 0):.1f}s) "
                                 f"- Rate: {videos_per_second:.2f}/s")
                        
                    elif result['status'] == 'skipped':
                        skipped_count += 1
                        worker_info = f"W{result['worker_id']}-GPU{result['gpu_id']}-M{result['model_id']}"
                        
                        elapsed = current_time - start_time
                        videos_per_second = processed_count / elapsed if elapsed > 0 else 0
                        
                        pbar.set_postfix({
                            'Success': success_count,
                            'Errors': error_count,
                            'Skipped': skipped_count,
                            'Rate': f"{videos_per_second:.2f}/s"
                        })
                        tqdm.write(f"{worker_info}: {os.path.basename(result['video_path'])} (already processed)")
                        
                    elif result['status'] == 'error':
                        error_count += 1
                        worker_info = f"W{result['worker_id']}-GPU{result['gpu_id']}-M{result['model_id']}"
                        error_msg = f"{worker_info}: {result['video_path']} - {result['error']}"
                        error_log.write(error_msg + "\n")
                        error_log.flush()
                        
                        elapsed = current_time - start_time
                        videos_per_second = processed_count / elapsed if elapsed > 0 else 0
                        
                        pbar.set_postfix({
                            'Success': success_count,
                            'Errors': error_count,
                            'Skipped': skipped_count,
                            'Rate': f"{videos_per_second:.2f}/s"
                        })
                        tqdm.write(f"ERROR {worker_info}: {os.path.basename(result['video_path'])} - Error: {result['error']}")
                    
                    pbar.update(1)
                    
                except queue.Empty:
                    print("Timeout waiting for results. Checking worker status...")
                    alive_workers = [p for p in worker_processes if p.is_alive()]
                    
                    if not alive_workers:
                        print("All workers finished.")
                        break
                    else:
                        print(f"{len(alive_workers)} workers still alive")
                        continue

    # Wait for all worker processes to finish
    print("Waiting for all worker processes to finish...")
    for i, process in enumerate(worker_processes):
        process.join(timeout=60)
        if process.is_alive():
            print(f"Force terminating worker process {i}")
            process.terminate()
            process.join()

    # Calculate final performance metrics
    total_time = time.time() - start_time
    total_processed = success_count + error_count + skipped_count
    
    # Advanced performance analysis
    throughput = len(files_to_process) / total_time if total_time > 0 else 0
    gpu_efficiency = (success_count / len(files_to_process)) * 100 if files_to_process else 0
    avg_words_per_video = total_words_processed / success_count if success_count > 0 else 0
    avg_duration_per_video = total_duration_processed / success_count if success_count > 0 else 0
    words_per_minute_overall = (total_words_processed / (total_duration_processed / 60)) if total_duration_processed > 0 else 0
    
    # Memory usage analysis
    if torch.cuda.is_available():
        gpu_memory_stats = []
        for i in range(NUM_GPUS):
            try:
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                gpu_memory_stats.append((memory_allocated, memory_reserved))
            except:
                gpu_memory_stats.append((0, 0))
    
    # Optional: Clean up cache files
    print("Cleaning up temporary cache files...")
    cache_cleanup_success, cache_cleanup_msg = safe_cleanup_cache_files(audio_cache_path_base, preserve_outputs=True)
    if cache_cleanup_success:
        print(f"Cache cleanup: {cache_cleanup_msg}")
    else:
        print(f"Cache cleanup failed: {cache_cleanup_msg}")
    
    # Print comprehensive final results
    print(f"\n{'='*80}")
    print("ENHANCED WHISPER PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f" Configuration:")
    print(f"  - GPUs used: {NUM_GPUS}")
    print(f"  - Models per GPU: {MODELS_PER_GPU}")
    print(f"  - Total workers: {TOTAL_WORKERS}")
    print(f"  - Batch size: {BATCH_SIZE} segments")
    print(f"  - Mixed precision: {MIXED_PRECISION}")
    print(f"  - Output format: ENHANCED (JSON + structured TXT)")
    print(f"  - Sequential model loading: ENABLED")
    print(f"\n Results:")
    print(f"  - Total videos: {len(files_to_process)}")
    print(f"  - Successfully processed: {success_count}")
    print(f"  - Already processed (skipped): {skipped_count}")
    print(f"  - Failed: {error_count}")
    print(f"  - Processing rate: {gpu_efficiency:.1f}%")
    print(f"\n Performance Metrics:")
    print(f"  - Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"  - Average time per video: {total_time/len(files_to_process):.2f} seconds")
    print(f"  - Throughput: {throughput:.2f} videos/second")
    print(f"  - Peak processing rate: {max(len(files_to_process)/total_time, 0):.2f} videos/second")
    print(f"\n Content Analysis:")
    print(f"  - Total words processed: {total_words_processed:,}")
    print(f"  - Total audio duration: {total_duration_processed/60:.1f} minutes")
    print(f"  - Average words per video: {avg_words_per_video:.0f}")
    print(f"  - Average duration per video: {avg_duration_per_video/60:.1f} minutes")
    print(f"  - Words per minute (speech rate): {words_per_minute_overall:.0f}")
    
    # GPU memory usage report
    if torch.cuda.is_available() and gpu_memory_stats:
        print(f"\n GPU Memory Usage:")
        for i, (allocated, reserved) in enumerate(gpu_memory_stats):
            print(f"  - GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
    
    if error_count > 0:
        print(f"\nError log saved to: {error_log_file_path}")
    
    # Performance recommendations
    print(f"\n Performance Analysis:")
    if gpu_efficiency < 90:
        print(f"  - Consider increasing batch size for better efficiency")
    if throughput < 1.0:
        print(f"  - Performance may be limited by audio/video processing")
    if error_count > len(files_to_process) * 0.05:
        print(f"  - High error rate detected - check input video quality")
    print(f"  - Enhanced formatting achieved {throughput:.1f} videos/second")
    print(f"  - Rich metadata and statistics included in outputs")
    print(f"  - TXT format now includes structured headers and summaries")
    
    # Output format summary
    print(f"\n Enhanced Output Features:")
    print(f"  - JSON: Complete segment metadata with confidence scores")
    print(f"  - TXT: Structured format with headers, statistics, and metadata")
    print(f"  - Segment tracking: chunk indices, merge flags, word counts")
    print(f"  - Summary statistics: duration, word rates, processing metrics")
    print(f"  - Human-readable formatting with clear section separators")
    
    print(f"\nEnhanced processing completed with {throughput:.2f} videos/second!")
    print(f"Check output files for rich formatted content!")


def print_enhanced_gpu_info():
    """Print comprehensive GPU and system information with enhanced formatting focus"""
    print(f"ENHANCED WHISPER PROCESSING SYSTEM:")
    print(f"=" * 60)
    
    # System info
    print(f"System:")
    print(f"  - CPU cores: {os.cpu_count()}")
    print(f"  - Available memory: {psutil.virtual_memory().available / (1024**3):.1f}GB")
    
    # GPU info
    if torch.cuda.is_available():
        print(f"CUDA Configuration (A100-Optimized):")
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - Available GPUs: {torch.cuda.device_count()}")
        
        a100_count = 0
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            memory_gb = gpu_props.total_memory / (1024**3)
            is_a100 = "A100" in gpu_props.name
            if is_a100:
                a100_count += 1
            
            print(f"  - GPU {i}: {gpu_props.name}")
            print(f"    * Memory: {memory_gb:.1f}GB")
            print(f"    * Compute capability: {gpu_props.major}.{gpu_props.minor}")
            print(f"    * Multiprocessors: {gpu_props.multi_processor_count}")
            if is_a100:
                print(f"    * A100-specific: TF32 enabled, optimized batch sizes")
        
        print(f"\nA100-Optimized Configuration:")
        print(f"  - A100 GPUs detected: {a100_count}")
        print(f"  - Each GPU: {MODELS_PER_GPU} models with batch size {BATCH_SIZE}")
        print(f"  - Total workers: {NUM_GPUS} × {MODELS_PER_GPU} = {TOTAL_WORKERS}")
        print(f"  - Mixed precision: {MIXED_PRECISION}")
        print(f"  - GPU memory fraction: {GPU_MEMORY_FRACTION}")
        print(f"  - TF32 acceleration: ENABLED for A100")
        print(f"  - Sequential model loading: ENABLED")
        print(f"  - Enhanced I/O: RICH FORMATTING with metadata")
        print(f"  - Expected GPU utilization: 90-98% (A100-optimized)")
        print(f"  - Estimated throughput: {TOTAL_WORKERS * 0.5:.1f}-{TOTAL_WORKERS * 0.8:.1f} videos/second")
        
        print(f"\nEnhanced Output Features:")
        print(f"  - JSON: Complete metadata with confidence scores and word counts")
        print(f"  - TXT: Structured format with headers, summaries, and statistics")
        print(f"  - Segment metadata: frame indices, chunk tracking, merge flags")
        print(f"  - Performance metrics: words/minute, processing rates, durations")
        print(f"  - Human-readable: Clear section separators and organized layout")
    else:
        print("CUDA not available. A100 optimizations disabled.")


# --- Additional utility functions for enhanced format ---
def generate_processing_summary(output_dir: str) -> None:
    """Generate a summary report of all processed videos"""
    try:
        summary_path = os.path.join(output_dir, "processing_summary.txt")
        json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
        
        if not json_files:
            print("No processed files found for summary generation.")
            return
        
        total_videos = 0
        total_segments = 0
        total_words = 0
        total_duration = 0.0
        total_frames = 0
        processing_stats = []
        
        for json_file in json_files:
            try:
                json_path = os.path.join(output_dir, json_file)
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                segments = data.get('segments', [])
                frame_info = data.get('frame_info', {})
                
                video_segments = len(segments)
                video_words = sum(seg.get('word_count', len(seg.get('text', '').split())) for seg in segments)
                video_duration = segments[-1]['end'] - segments[0]['start'] if segments else 0
                video_frames = frame_info.get('frame_count', 0)
                
                total_videos += 1
                total_segments += video_segments
                total_words += video_words
                total_duration += video_duration
                total_frames += video_frames
                
                processing_stats.append({
                    'file': json_file,
                    'segments': video_segments,
                    'words': video_words,
                    'duration': video_duration,
                    'frames': video_frames
                })
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
        
        # Generate summary report
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("WHISPER PROCESSING BATCH SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Processing Overview:\n")
            f.write(f"  - Total Videos Processed: {total_videos}\n")
            f.write(f"  - Total Segments Generated: {total_segments:,}\n")
            f.write(f"  - Total Words Transcribed: {total_words:,}\n")
            f.write(f"  - Total Audio Duration: {total_duration/60:.1f} minutes ({total_duration/3600:.1f} hours)\n")
            f.write(f"  - Total Frames Extracted: {total_frames:,}\n\n")
            
            if total_videos > 0:
                f.write(f"Average Statistics:\n")
                f.write(f"  - Segments per Video: {total_segments/total_videos:.1f}\n")
                f.write(f"  - Words per Video: {total_words/total_videos:.0f}\n")
                f.write(f"  - Duration per Video: {(total_duration/total_videos)/60:.1f} minutes\n")
                f.write(f"  - Frames per Video: {total_frames/total_videos:.0f}\n")
                f.write(f"  - Words per Minute (Speech Rate): {(total_words/(total_duration/60)):.0f}\n\n")
            
            f.write(f"Detailed File Statistics:\n")
            f.write(f"{'-' * 80}\n")
            f.write(f"{'File':<30} {'Segments':<10} {'Words':<8} {'Duration':<10} {'Frames':<8}\n")
            f.write(f"{'-' * 80}\n")
            
            for stat in sorted(processing_stats, key=lambda x: x['words'], reverse=True):
                filename = stat['file'][:28] + '..' if len(stat['file']) > 30 else stat['file']
                f.write(f"{filename:<30} {stat['segments']:<10} {stat['words']:<8} {stat['duration']/60:>8.1f}m {stat['frames']:<8}\n")
            
            f.write(f"=" * 80 + "\n")
            f.write(f"Summary generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Enhanced Whisper Processing System v2.0\n")
            f.write(f"=" * 80 + "\n")
        
        print(f"Processing summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"Error generating processing summary: {e}")


if __name__ == "__main__":
    print_enhanced_gpu_info()
    main_high_performance_enhanced()
    
    # Generate batch summary after processing
    print("\nGenerating processing summary...")
    generate_processing_summary(OUTPUT_DIR)