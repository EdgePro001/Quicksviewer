"""
QwenFilterLocal.py

Download Qwen2.5-VL-72B-Instruct Model to cluster, evaluate similarity between each frame in a single 20s-long segment and text prompt. Keep segments with similarity higher than threshold(default = 0.75).

To be updated: 
1) Use vllm to optimize, instead of downloading model.
2) Model parallel for higher performance.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
from typing import List, Dict, Tuple, Optional
import re
import glob
from tqdm import tqdm
import argparse
from pathlib import Path
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from modelscope import snapshot_download
from PIL import Image
import os

class QwenFilter:
    def __init__(self, model_dir: str = None, model=None, processor=None, threshold: float = 0.75, device: str = "auto"):
        """
        Args:
            model_dir: Directory of Loaded Model
            threshold: Similarity threshold
            device: cuda or cpu 
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        try:
            print("Downloading/Loading model from ModelScope...")
            model_dir = snapshot_download('qwen/Qwen2.5-VL-7B-Instruct')
            print(f"Model available at: {model_dir}")

            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_dir, 
                torch_dtype=torch.bfloat16, 
                attn_implementation="eager",  # Use standard attention instead of flash_attention_2
                device_map="auto",
                trust_remote_code=True
            )
            processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

            self.model = model
            self.processor = processor

            print("Model loaded successfully!")
        except Exception as e:
            print(f"Model loading failed: {e}")
            print("Trying alternative loading method...")
            
            try:
                # Fallback: try to load from HuggingFace directly (without flash_attention)
                model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path, 
                    torch_dtype=torch.bfloat16, 
                    attn_implementation="eager",  # Use standard attention
                    device_map="auto",
                    trust_remote_code=True
                )
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                
                self.model = model
                self.processor = processor
                print("Model loaded successfully with HuggingFace fallback method!")
            except Exception as e2:
                print(f"All loading methods failed: {e2}")
                raise e2
        
        self.threshold = threshold

    def getFrames(self, frames_dir: str, beg: float, end: float) -> List[Tuple[str, float]]:
        """
        Args:
            frames_dir: Directory of cached frames
            beg: start time of each 20s segment
            end: end time of each segment

        Returns:
            A list of Tuples(Path to frame file, Timestamp)
        """
        if not os.path.exists(frames_dir):
            return []
        
        frame_files = glob.glob(os.path.join(frames_dir, "frame_*.jpg"))
        if not frame_files:
            return []
        
        frames_in_range = []
        
        for frame_file in frame_files:
            filename = os.path.basename(frame_file)
            # matching format: frame_000001_t5.00.jpg
            match = re.search(r'frame_(\d+)_t([\d.]+)\.jpg', filename)
            if match:
                timestamp = float(match.group(2))
                if beg <= timestamp < end:
                    frames_in_range.append((frame_file, timestamp))
        
        # Sort with timestamps
        frames_in_range.sort(key=lambda x: x[1])
        
        return frames_in_range

    def getSimilarity(self, image_dir, prompt: str, model, processor, max_new_tokens = 512):
        """
        Use model to evaluate similarity between image and prompt

        Args:
            image_dir: path to image
            prompt: Transcription of each 20s segments
            model: Loaded model (Default Qwen2.5VL-7B-Instruct)
            processor: Loaded Processor
            max_new_tokens: Maximum Tokens to generate

        Returns:
            A float of similarity score
            e.g. 0.860
        """

        image = Image.open(image_dir)
        image_localPath = "file://" + image_dir

        sys_prompt = """
        You are an expert video-text alignment evaluator. Your task is to assess how well the transcribed text describes what is actually happening in the image frame.
            Evaluation criteria:
            1. Semantic Relevance: Does the text accurately describe the visual content, actions, or events shown in the image?
            2. Temporal Alignment: Is this text likely to correspond to the specific moment captured in this frame?
            3. Content Quality: Is the text coherent and meaningful, not just random words or garbled speech?

            Scoring guidelines:
            - 0.8-1.0: Text precisely describes the visual content with clear temporal alignment
            - 0.6-0.8: Text is generally relevant but may have minor misalignments or unclear references  
            - 0.4-0.6: Text has some connection to the image but significant gaps or inaccuracies
            - 0.2-0.4: Text barely relates to the visual content or is mostly irrelevant
            - 0.0-0.2: Text is completely unrelated, nonsensical, or appears to be transcription errors

            Important: Be strict with your scoring. Only give high scores (>0.8) when there is clear evidence that the text genuinely describes what's happening in the image at that specific moment.

            Provide only a single decimal number between 0 and 1 (e.g., 0.635).
        """

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content":[
                {"type": "text", "text": prompt},
                {"image": image_localPath},
            ]},
        ]

        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Process inputs
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)

        # Generate output
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        # Decode output
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        result = output_text[0].strip()
        
        # Try to convert to float
        try:
            return float(result)
        except ValueError:
            # If conversion fails, try to extract number from string
            import re
            numbers = re.findall(r'\d*\.?\d+', result)
            if numbers:
                return float(numbers[0])
            else:
                return 0.0

    def getSegmentSimilarity(self, segment: Dict, frames_dir: str) -> Tuple[float, str]:
        """
        Getting the highest score of each segment

        Args:
            segment: Information of each segment
            frames_dir: Directory to frame images

        Returns:
            a tuple(Maximum of Similarity score(float), path to best matched image)
        """
        text = segment.get('text', '').strip()
        if not text:
            return 0.0, None
        
        frames = self.getFrames(
            frames_dir, 
            segment['start'], 
            segment['end']
        )
        
        if not frames:
            return 0.0, None
        
        max_similarity = 0.0
        best_frame_path = None
        
        for frame_path, timestamp in frames:
            similarity = self.getSimilarity(frame_path, text, self.model, self.processor)
            if similarity > max_similarity:
                max_similarity = similarity
                best_frame_path = frame_path
        
        return max_similarity, best_frame_path
        
    def filterSegments(self, json_path: str, output_path: str, frames_dir: str = None) -> Dict:
        """
        Filtering Segments in JSON file, get segments which have high similarity.

        Args:
            json_path: path to JSON file of segments
            output_path: path to output filtered JSON file
            frames_dir: path to frames directory

        Returns:
            A Dict of stats
        """

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        segments = data.get('segments', [])
        frame_info = data.get('frame_info', {})

        if frames_dir is None:
            frames_dir = frame_info.get('frame_dir', '')
        if not segments:
            print("No segments found in JSON file")
            return {}
        
        filtered_segments = []
        similarity_scores = []

        print(f"Processing {len(segments)} segments...")
        for segment in tqdm(segments, desc="Filtering segments"):
            max_similarity, best_frame = self.getSegmentSimilarity(segment, frames_dir)
            if max_similarity > 0:
                similarity_scores.append(max_similarity)

                if max_similarity >= self.threshold:
                    filtered_segment = segment.copy()
                    filtered_segment["similarity_score"] = max_similarity
                    filtered_segment["best_frame"] = best_frame
                    filtered_segments.append(filtered_segment)
        
        output_data = {
            'segments': filtered_segments,
            'frame_info': frame_info,
            'metadata': {
                **data.get('metadata', {}),
                'filtering': {
                    'similarity_threshold': self.threshold,
                    'original_segments_count': len(segments),
                    'filtered_segments_count': len(filtered_segments),
                    'average_similarity': round(np.mean(similarity_scores), 4) if similarity_scores else 0,
                    'max_similarity': round(np.max(similarity_scores), 4) if similarity_scores else 0,
                    'min_similarity': round(np.min(similarity_scores), 4) if similarity_scores else 0
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
                        
    def processVideos(self, input_dir: str, output_dir: str):
        """
        Processing all JSONs and Videos of certain Path
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # find all jsons available 
        json_files = list(input_path.rglob("*_aligned.json"))
        if not json_files:
            print(f"No aligned JSON files found in {input_dir}")
            return {}
        
        print(f"Found {len(json_files)} JSON files to process")

        # Initialize Total Stats
        total_stats = {
            'total_files': len(json_files),
            'processed_files': 0,
            'total_segments': 0,
            'filtered_segments': 0,
            'average_similarity': []
        }
        
        # Process each JSON file
        for json_file in tqdm(json_files, desc="Processing videos"):
            try:
                # Construct relative path for output
                relative_path = json_file.relative_to(input_path)
                output_json_path = output_path / relative_path
                
                # Create output directory
                output_json_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Get corresponding frames directory
                video_name = json_file.stem.replace('_aligned', '')
                
                # Build frames directory path based on JSON file structure
                # JSON: /user/majunxian/whisper/output_20s/external-data-youtube/video/20241103/00/6QC1NQeY26o_aligned.json
                # Frames: /user/majunxian/whisper/output_20s/external-data-youtube/video/20241103/.frames_cache/00/6QC1NQeY26o/
                
                json_parent = json_file.parent  # .../00/
                date_parent = json_parent.parent  # .../20241103/
                
                # Construct frames path: date_parent/.frames_cache/hour_dir/video_name/
                frames_dir = date_parent / '.frames_cache' / json_parent.name / video_name
                
                if not frames_dir.exists():
                    print(f"Warning: No frames directory found for {video_name} at {frames_dir}")
                    continue
                
                frames_dir = str(frames_dir)
                
                # Process single file
                stats = self.filterSegments(
                    json_path=str(json_file),
                    output_path=str(output_json_path),
                    frames_dir=frames_dir
                )
                
                total_stats['processed_files'] += 1
                total_stats['total_segments'] += stats.get('original_count', 0)
                total_stats['filtered_segments'] += stats.get('filtered_count', 0)
                if stats.get('average_similarity', 0) > 0:
                    total_stats['average_similarity'].append(stats['average_similarity'])
                
                print(f"Processed {json_file.name}: "
                      f"{stats.get('filtered_count', 0)}/{stats.get('original_count', 0)} segments retained "
                      f"(avg similarity: {stats.get('average_similarity', 0):.3f})")
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
        
        # Calculate overall statistics
        if total_stats['average_similarity']:
            total_stats['overall_average_similarity'] = np.mean(total_stats['average_similarity'])
        else:
            total_stats['overall_average_similarity'] = 0
        
        print(f"\n--- Overall Processing Results ---")
        print(f"Total files: {total_stats['total_files']}")
        print(f"Processed files: {total_stats['processed_files']}")
        print(f"Total segments: {total_stats['total_segments']}")
        print(f"Filtered segments: {total_stats['filtered_segments']}")
        if total_stats['total_segments'] > 0:
            print(f"Overall retention rate: {total_stats['filtered_segments'] / total_stats['total_segments']:.2%}")
        print(f"Overall average similarity: {total_stats['overall_average_similarity']:.3f}")
        print(f"Similarity threshold: {self.threshold}")
        
        return total_stats

def main():
    parser = argparse.ArgumentParser(description="Filter video segments using Qwen2.5-VL image-text similarity")
    parser.add_argument("--input_dir", required=True, help="Input directory containing JSON files and frames")
    parser.add_argument("--output_dir", default="./filtered_output", help="Output directory for filtered JSON files")
    parser.add_argument("--threshold", type=float, default=0.75, 
                       help="Similarity threshold (0-1)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} not found")
        return
    
    # Create filter
    try:
        filter_tool = QwenFilter(
            threshold=args.threshold,
            device=args.device
        )
    except Exception as e:
        print(f"Error initializing QwenFilter: {e}")
        return
    
    # Execute filtering
    stats = filter_tool.processVideos(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
