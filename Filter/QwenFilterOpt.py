"""
QwenFilterOpt.py - GPU 4-7 Specialized Version with Continuous Monitoring

Use vLLM API server with parallel processing for Qwen2.5-VL-72B-Instruct Model 
to evaluate similarity between frames and text prompts.

Optimized for GPU 4-7 usage and Whisper Pool results processing with continuous monitoring.
"""

import os
import json
import base64
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Set
import re
import glob
from tqdm import tqdm
import argparse
from pathlib import Path
from openai import OpenAI
import concurrent.futures
import threading
import time
from functools import partial
import signal
import sys
from datetime import datetime

class QwenFilterGPU47:
    def __init__(self, threshold: float = 0.6, api_base: str = "http://localhost:8001/v1", 
                 api_key: str = "EMPTY", model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct",
                 max_workers: int = 8, max_concurrent_requests: int = 16):
        """
        GPU 4-7 specialized version for Whisper Pool results filtering with continuous monitoring
        
        Args:
            threshold: Similarity threshold
            api_base: vLLM API server base URL (default port 8001 for GPU 4-7)
            api_key: API key (use "EMPTY" for local vLLM server)
            model_name: Model name as configured in vLLM server
            max_workers: Maximum number of worker threads
            max_concurrent_requests: Maximum concurrent API requests
        """
        self.threshold = threshold
        self.api_base = api_base
        self.model_name = model_name
        self.max_workers = max_workers
        self.max_concurrent_requests = max_concurrent_requests
        
        # Create a semaphore to limit concurrent requests
        self.semaphore = threading.Semaphore(max_concurrent_requests)
        
        # Initialize OpenAI client for vLLM API
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=45.0
        )
        
        # Continuous monitoring state
        self.processed_files: Set[str] = set()
        self.processing_log_path: Optional[str] = None
        self.stop_monitoring = False
        
        # Test connection to vLLM server
        self._test_connection()
        
        # Pre-compile regex for performance
        self.frame_pattern = re.compile(r'frame_(\d+)_t([\d.]+)\.jpg')
        
        print(f"QwenFilter initialized for GPU 4-7:")
        print(f"  - API Base: {api_base}")
        print(f"  - Max Workers: {max_workers}")
        print(f"  - Max Concurrent Requests: {max_concurrent_requests}")
        print(f"  - Similarity Threshold: {threshold}")
        
    def _test_connection(self):
        """Test if vLLM server is running and accessible on GPU 4-7"""
        try:
            models = self.client.models.list()
            print(f"Connected to GPU 4-7 vLLM server. Available models: {[m.id for m in models.data]}")
        except Exception as e:
            print(f"Failed to connect to GPU 4-7 vLLM server at {self.api_base}")
            print(f"Error: {e}")
            print(f"Please make sure GPU 4-7 vLLM server is running on port 8001")
            raise

    def _load_processed_files(self, log_path: str):
        """Load list of already processed files from log"""
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    data = json.load(f)
                    self.processed_files = set(data.get('processed_files', []))
                    print(f"Loaded {len(self.processed_files)} processed files from log")
            except Exception as e:
                print(f"Error loading processed files log: {e}")
                self.processed_files = set()
        else:
            self.processed_files = set()
            
    def _save_processed_files(self, log_path: str):
        """Save list of processed files to log"""
        try:
            log_data = {
                'processed_files': list(self.processed_files),
                'last_update': datetime.now().isoformat(),
                'total_processed': len(self.processed_files)
            }
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            print(f"Error saving processed files log: {e}")

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string for API"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            raise

    def getFrames(self, frames_dir: str, beg: float, end: float) -> List[Tuple[str, float]]:
        """
        Get frames from Whisper cache directory
        """
        if not os.path.exists(frames_dir):
            return []
        
        frame_files = glob.glob(os.path.join(frames_dir, "frame_*.jpg"))
        if not frame_files:
            return []
        
        frames_in_range = []
        
        for frame_file in frame_files:
            filename = os.path.basename(frame_file)
            match = self.frame_pattern.search(filename)
            if match:
                timestamp = float(match.group(2))
                if beg <= timestamp < end:
                    frames_in_range.append((frame_file, timestamp))
        
        frames_in_range.sort(key=lambda x: x[1])
        return frames_in_range

    def getSimilarity(self, image_path: str, prompt: str, max_tokens: int = 10) -> float:
        """
        Use GPU 4-7 vLLM API to evaluate similarity between image and prompt
        """
        # Quick text filtering
        text_words = prompt.strip().split()
        if len(text_words) < 3:
            return 0.1
            
        # Use semaphore to limit concurrent requests
        with self.semaphore:
            try:
                base64_image = self._encode_image_to_base64(image_path)
            except Exception as e:
                print(f"Error encoding image {image_path}: {e}")
                return 0.0

            sys_prompt = """
You are a strict video-text alignment evaluator for Whisper transcription results. Your task is to assess whether the Whisper-transcribed text genuinely describes the visual content observable in this specific image frame.


LANGUAGE REQUIREMENT:
- **ENGLISH ONLY**: This evaluation system only processes English text. Any text that is not in English should automatically receive a score of 0.0-0.2, regardless of visual alignment.
- **NON-ENGLISH AUTOMATIC LOW SCORES**: Text in Chinese, Spanish, French, German, Japanese, Korean, Arabic, Russian, or any other non-English language should be scored 0.0-0.2.
- **MIXED LANGUAGE**: Text containing significant non-English words or phrases should receive low scores (0.1-0.3).

EVALUATION CRITERIA (ALL must be met for high scores):
1. VISUAL EVIDENCE: The text must describe something you can actually see in the image
2. SPECIFICITY: Generic phrases, emotions, or dialogue rarely warrant high scores
3. TEMPORAL PRECISION: The text should match this exact moment, not just the general video topic
4. DESCRIPTIVE NATURE: Prioritize descriptive content over conversational/dialogue content

STRICT SCORING GUIDELINES:
- 0.9-1.0: Text describes specific visible elements (objects, actions, scenes, people, movements) that are clearly present in this frame
- 0.7-0.8: Text partially describes visible content but may include some non-visual elements
- 0.5-0.6: Text has loose connection to visible content but is mostly non-descriptive
- 0.3-0.4: Text is generic dialogue/conversation that could apply to any similar context
- 0.1-0.2: Text is pure dialogue, emotions, or reactions with no visual description
- 0.0: Text is completely unrelated, nonsensical, or transcription errors

AUTOMATIC LOW SCORES (0.0-0.3) for:
- **SHORT TEXT (under 5 words)**: Automatically score 0.0-0.2 regardless of content
- **FRAGMENTS**: Single words, incomplete sentences, or ellipses ("I", "...", "Yeah", "What?")
- Pure dialogue without visual context ("Thank you", "I said...", "What do you think?")
- Emotional expressions only ("Wow", "Oh no", "Amazing")
- Generic conversation that doesn't describe visuals
- Background conversation unrelated to what's shown
- Common Whisper transcription artifacts ("[Music]", "[Applause]", etc.)

CRITICAL: Text with fewer than 5 words should almost never score above 0.2, even if seemingly relevant.

HIGH SCORES only when text describes:
- Specific objects, people, or animals visible in the frame
- Actions or movements happening in the image
- Scene settings or environments clearly shown
- Visual changes or events occurring in this moment
- Specific visual details that match the frame content

Be extremely strict. Most conversational Whisper transcripts should score below 0.5. Only descriptive content about visible elements deserves scores above 0.7.

Provide only a single decimal number between 0.000 and 1.000 (e.g., 0.235).
            """

            messages = [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": f"Whisper transcribed text to evaluate: {prompt}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    top_p=1.0
                )
                
                result = response.choices[0].message.content.strip()
                
                try:
                    score = float(result)
                    return max(0.0, min(1.0, score))
                except ValueError:
                    numbers = re.findall(r'\d*\.?\d+', result)
                    if numbers:
                        score = float(numbers[0])
                        return max(0.0, min(1.0, score))
                    else:
                        return 0.0
                        
            except Exception as e:
                print(f"Error calling GPU 4-7 vLLM API: {e}")
                return 0.0

    def _process_single_frame(self, frame_data: Tuple[str, float], text: str) -> Tuple[float, str, float]:
        """Process a single frame for similarity"""
        frame_path, timestamp = frame_data
        similarity = self.getSimilarity(frame_path, text)
        return similarity, frame_path, timestamp

    def getSegmentSimilarity(self, segment: Dict, frames_dir: str) -> Tuple[float, str]:
        """
        Getting the highest score of each Whisper segment using parallel processing
        """
        text = segment.get('text', '').strip()
        if not text:
            return 0.0, None
        
        frames = self.getFrames(frames_dir, segment['start'], segment['end'])
        
        if not frames:
            return 0.0, None
        
        max_similarity = 0.0
        best_frame_path = None
        
        process_frame_func = partial(self._process_single_frame, text=text)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(frames))) as executor:
            future_to_frame = {executor.submit(process_frame_func, frame): frame for frame in frames}
            
            for future in concurrent.futures.as_completed(future_to_frame):
                try:
                    similarity, frame_path, timestamp = future.result()
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_frame_path = frame_path
                        
                    # Early stopping if we find a very good match
                    if similarity > 0.9:
                        for remaining_future in future_to_frame:
                            if remaining_future != future:
                                remaining_future.cancel()
                        break
                        
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue
        
        return max_similarity, best_frame_path

    def _process_single_segment(self, segment_data: Tuple[Dict, str]) -> Tuple[Dict, float, str]:
        """Process a single Whisper segment"""
        segment, frames_dir = segment_data
        max_similarity, best_frame = self.getSegmentSimilarity(segment, frames_dir)
        return segment, max_similarity, best_frame
        
    def filterWhisperSegments(self, json_path: str, output_path: str, frames_dir: str = None) -> Dict:
        """
        Filtering Whisper Segments in JSON file using parallel processing
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        segments = data.get('segments', [])
        frame_info = data.get('frame_info', {})

        # Auto-detect frames directory from Whisper cache structure
        if frames_dir is None:
            json_name = Path(json_path).stem
            pool_dir = Path(json_path).parent
            frames_cache_dir = pool_dir / "frames_cache" / json_name
            if frames_cache_dir.exists():
                frames_dir = str(frames_cache_dir)
            else:
                frames_dir = frame_info.get('frame_dir', '')
                
        if not segments:
            print("No Whisper segments found in JSON file")
            return {}
        
        filtered_segments = []
        similarity_scores = []

        print(f"Processing {len(segments)} Whisper segments in parallel on GPU 4-7...")
        
        segment_data_list = [(segment, frames_dir) for segment in segments]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_segment = {
                executor.submit(self._process_single_segment, segment_data): segment_data[0] 
                for segment_data in segment_data_list
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_segment), 
                             total=len(future_to_segment), desc="Filtering Whisper segments on GPU 4-7"):
                try:
                    segment, max_similarity, best_frame = future.result()
                    
                    if max_similarity > 0:
                        similarity_scores.append(max_similarity)

                        if max_similarity >= self.threshold:
                            filtered_segment = segment.copy()
                            filtered_segment["similarity_score"] = max_similarity
                            filtered_segment["best_frame"] = best_frame
                            filtered_segments.append(filtered_segment)
                            
                except Exception as e:
                    print(f"Error processing Whisper segment: {e}")
                    continue
        
        output_data = {
            'segments': filtered_segments,
            'frame_info': frame_info,
            'metadata': {
                **data.get('metadata', {}),
                'filtering': {
                    'processor': 'QwenFilter_GPU4_7',
                    'similarity_threshold': self.threshold,
                    'original_segments_count': len(segments),
                    'filtered_segments_count': len(filtered_segments),
                    'retention_rate': len(filtered_segments) / len(segments) if segments else 0,
                    'average_similarity': round(np.mean(similarity_scores), 4) if similarity_scores else 0,
                    'max_similarity': round(np.max(similarity_scores), 4) if similarity_scores else 0,
                    'min_similarity': round(np.min(similarity_scores), 4) if similarity_scores else 0,
                    'processing_gpu': '4-7',
                    'api_endpoint': self.api_base,
                    'processing_time': datetime.now().isoformat()
                }
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        stats = {
            'original_count': len(segments),
            'filtered_count': len(filtered_segments),
            'retention_rate': len(filtered_segments) / len(segments) if segments else 0,
            'average_similarity': np.mean(similarity_scores) if similarity_scores else 0,
            'similarity_threshold': self.threshold
        }
        
        return stats

    def _get_new_json_files(self, input_dir: str) -> List[Path]:
        """Get list of new JSON files that haven't been processed yet"""
        input_path = Path(input_dir)
        all_json_files = list(input_path.glob("*.json"))
        
        new_files = []
        for json_file in all_json_files:
            if json_file.name not in self.processed_files:
                new_files.append(json_file)
        
        return new_files

    def processWhisperResults(self, input_dir: str, output_dir: str) -> Dict:
        """
        Processing all Whisper JSON results from Pool directory (single batch)
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        new_json_files = self._get_new_json_files(input_dir)
        
        if not new_json_files:
            print("No new JSON files found to process")
            return {}
        
        print(f"Found {len(new_json_files)} new Whisper JSON files to process on GPU 4-7")

        batch_stats = {
            'batch_files': len(new_json_files),
            'processed_files': 0,
            'total_segments': 0,
            'filtered_segments': 0,
            'average_similarity': []
        }
        
        def process_single_whisper_json(json_file):
            """Process a single Whisper JSON file"""
            try:
                output_json_path = output_path / f"filtered_{json_file.name}"
                
                video_name = json_file.stem
                frames_dir = input_path / "frames_cache" / video_name
                
                if not frames_dir.exists():
                    print(f"Warning: No frames directory found for {video_name} at {frames_dir}")
                    frames_dir = input_path / ".." / "frames_cache" / video_name
                    if not frames_dir.exists():
                        print(f"Warning: No frames found for {video_name}, skipping")
                        return None
                
                frames_dir = str(frames_dir)
                
                stats = self.filterWhisperSegments(
                    json_path=str(json_file),
                    output_path=str(output_json_path),
                    frames_dir=frames_dir
                )
                
                # Mark file as processed
                self.processed_files.add(json_file.name)
                
                return {
                    'file': json_file.name,
                    'stats': stats
                }
                
            except Exception as e:
                print(f"Error processing Whisper file {json_file}: {e}")
                return None
        
        max_file_workers = min(3, len(new_json_files))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_file_workers) as executor:
            future_to_file = {executor.submit(process_single_whisper_json, json_file): json_file 
                            for json_file in new_json_files}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                             total=len(future_to_file), desc="Processing new Whisper files on GPU 4-7"):
                result = future.result()
                if result:
                    stats = result['stats']
                    batch_stats['processed_files'] += 1
                    batch_stats['total_segments'] += stats.get('original_count', 0)
                    batch_stats['filtered_segments'] += stats.get('filtered_count', 0)
                    if stats.get('average_similarity', 0) > 0:
                        batch_stats['average_similarity'].append(stats['average_similarity'])
                    
                    print(f"GPU 4-7 processed {result['file']}: "
                          f"{stats.get('filtered_count', 0)}/{stats.get('original_count', 0)} segments retained "
                          f"(avg similarity: {stats.get('average_similarity', 0):.3f})")
        
        if batch_stats['average_similarity']:
            batch_stats['overall_average_similarity'] = np.mean(batch_stats['average_similarity'])
        else:
            batch_stats['overall_average_similarity'] = 0
        
        # Save processed files log
        if self.processing_log_path:
            self._save_processed_files(self.processing_log_path)
        
        return batch_stats

    def continuousMonitoring(self, input_dir: str, output_dir: str, 
                           check_interval: int = 30, max_idle_time: int = 18000) -> Dict:
        """
        Continuously monitor Pool directory for new JSON files and process them
        
        Args:
            input_dir: Whisper Pool directory to monitor
            output_dir: Output directory for filtered results
            check_interval: Interval in seconds to check for new files
            max_idle_time: Maximum time in seconds to wait without new files before stopping
        """
        print(f"Starting continuous monitoring of {input_dir}")
        print(f"Check interval: {check_interval}s, Max idle time: {max_idle_time}s")
        
        # Initialize processing log
        self.processing_log_path = os.path.join(output_dir, "processed_files.json")
        self._load_processed_files(self.processing_log_path)
        
        total_stats = {
            'monitoring_start': datetime.now().isoformat(),
            'total_batches': 0,
            'total_files_processed': 0,
            'total_segments': 0,
            'total_filtered_segments': 0,
            'batch_details': []
        }
        
        last_activity_time = time.time()
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}, stopping monitoring...")
            self.stop_monitoring = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            while not self.stop_monitoring:
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking for new JSON files...")
                
                # Check for new files
                new_files = self._get_new_json_files(input_dir)
                
                if new_files:
                    print(f"Found {len(new_files)} new files to process")
                    last_activity_time = time.time()
                    
                    # Process new files
                    batch_start_time = time.time()
                    batch_stats = self.processWhisperResults(input_dir, output_dir)
                    batch_end_time = time.time()
                    
                    if batch_stats:
                        batch_info = {
                            'batch_number': total_stats['total_batches'] + 1,
                            'timestamp': datetime.now().isoformat(),
                            'processing_time': batch_end_time - batch_start_time,
                            **batch_stats
                        }
                        
                        total_stats['total_batches'] += 1
                        total_stats['total_files_processed'] += batch_stats.get('processed_files', 0)
                        total_stats['total_segments'] += batch_stats.get('total_segments', 0)
                        total_stats['total_filtered_segments'] += batch_stats.get('filtered_segments', 0)
                        total_stats['batch_details'].append(batch_info)
                        
                        print(f"\nBatch {total_stats['total_batches']} completed:")
                        print(f"  - Files processed: {batch_stats.get('processed_files', 0)}")
                        print(f"  - Processing time: {batch_end_time - batch_start_time:.1f}s")
                        print(f"  - Retention rate: {batch_stats.get('filtered_segments', 0) / batch_stats.get('total_segments', 1):.2%}")
                        
                        # Save monitoring stats
                        monitoring_log_path = os.path.join(output_dir, "monitoring_stats.json")
                        try:
                            with open(monitoring_log_path, 'w') as f:
                                json.dump(total_stats, f, indent=2)
                        except Exception as e:
                            print(f"Error saving monitoring stats: {e}")
                            
                else:
                    idle_time = time.time() - last_activity_time
                    print(f"No new files found. Idle time: {idle_time:.0f}s / {max_idle_time}s")
                    
                    # Check if we've been idle too long
                    if idle_time > max_idle_time:
                        print(f"Maximum idle time ({max_idle_time}s) reached. Stopping monitoring.")
                        break
                
                # Wait before next check
                print(f"Waiting {check_interval}s before next check...")
                for i in range(check_interval):
                    if self.stop_monitoring:
                        break
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            print("\nMonitoring interrupted by user")
        except Exception as e:
            print(f"Error during monitoring: {e}")
        finally:
            total_stats['monitoring_end'] = datetime.now().isoformat()
            total_stats['total_monitoring_time'] = time.time() - last_activity_time
            
            print(f"\n--- Continuous Monitoring Summary ---")
            print(f"Total batches processed: {total_stats['total_batches']}")
            print(f"Total files processed: {total_stats['total_files_processed']}")
            print(f"Total segments processed: {total_stats['total_segments']}")
            print(f"Total filtered segments: {total_stats['total_filtered_segments']}")
            if total_stats['total_segments'] > 0:
                overall_retention = total_stats['total_filtered_segments'] / total_stats['total_segments']
                print(f"Overall retention rate: {overall_retention:.2%}")
            
            # Final save of stats
            try:
                monitoring_log_path = os.path.join(output_dir, "monitoring_stats.json")
                with open(monitoring_log_path, 'w') as f:
                    json.dump(total_stats, f, indent=2)
                print(f"Monitoring statistics saved to: {monitoring_log_path}")
            except Exception as e:
                print(f"Error saving final monitoring stats: {e}")
                
        return total_stats

def main():
    parser = argparse.ArgumentParser(description="Filter Whisper segments using Qwen2.5-VL on GPU 4-7")
    parser.add_argument("--input_dir", required=True, 
                       help="Whisper Pool directory containing JSON files and frames")
    parser.add_argument("--output_dir", default="/user/majunxian/SFTPipeline/FiltedOutput", 
                       help="Output directory for filtered JSON files")
    parser.add_argument("--threshold", type=float, default=0.6, 
                       help="Similarity threshold (0-1)")
    parser.add_argument("--api_base", default="http://localhost:8001/v1",
                       help="vLLM API server base URL (GPU 4-7, port 8001)")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-72B-Instruct",
                       help="Model name as configured in vLLM")
    parser.add_argument("--max_workers", type=int, default=8,
                       help="Maximum number of worker threads")
    parser.add_argument("--max_concurrent_requests", type=int, default=16,
                       help="Maximum concurrent API requests")
    
    # Continuous monitoring options
    parser.add_argument("--continuous", action="store_true",
                       help="Enable continuous monitoring mode")
    parser.add_argument("--check_interval", type=int, default=30,
                       help="Interval in seconds to check for new files (continuous mode)")
    parser.add_argument("--max_idle_time", type=int, default=18000,
                       help="Maximum time in seconds to wait without new files before stopping (continuous mode)")
    
    args = parser.parse_args()
    
    # Check Whisper Pool directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Whisper Pool directory {args.input_dir} not found")
        print("Please ensure Whisper processing has completed and Pool directory exists")
        return
    
    try:
        filter_tool = QwenFilterGPU47(
            threshold=args.threshold,
            api_base=args.api_base,
            model_name=args.model_name,
            max_workers=args.max_workers,
            max_concurrent_requests=args.max_concurrent_requests
        )
    except Exception as e:
        print(f"Error initializing QwenFilterGPU47: {e}")
        print("Please ensure GPU 4-7 vLLM server is running on port 8001")
        return
    
    start_time = time.time()
    
    if args.continuous:
        print(f"Starting continuous monitoring mode...")
        stats = filter_tool.continuousMonitoring(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            check_interval=args.check_interval,
            max_idle_time=args.max_idle_time
        )
    else:
        print(f"Starting single batch processing...")
        json_count = len(list(Path(args.input_dir).glob("*.json")))
        if json_count == 0:
            print(f"No JSON files found in {args.input_dir}")
            return
        
        print(f"Found {json_count} Whisper JSON files to process")
        stats = filter_tool.processWhisperResults(
            input_dir=args.input_dir,
            output_dir=args.output_dir
        )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\nProcessing completed in {processing_time:.1f} seconds")
    
    if not args.continuous and stats:
        print(f"Average processing speed: {stats.get('processed_files', 0) / processing_time * 60:.1f} files/minute")

if __name__ == "__main__":
    main()