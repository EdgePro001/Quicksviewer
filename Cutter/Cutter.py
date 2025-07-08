"""
Cutter.py

Video Segment Cutter
Extract video segments based on timestamp information in JSON files
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoSegmentCutter:
    def __init__(self, video_base_path: str, output_base_path: str, extend_seconds: int = 5, skip_missing: bool = False):
        """
        Initialize video segment cutter
        
        Args:
            video_base_path: Base path for video files
            output_base_path: Base path for output files
            extend_seconds: Number of seconds to extend on each side, default 5
            skip_missing: Skip missing video files instead of stopping, default False
        """
        self.video_base_path = Path(video_base_path)
        self.output_base_path = Path(output_base_path)
        self.extend_seconds = extend_seconds
        self.skip_missing = skip_missing
        self._segments_processed = 0  # Track processed segments
        
        # Create output directory
        self.output_base_path.mkdir(parents=True, exist_ok=True)
    
    def find_json_files(self, search_path: str) -> List[Path]:
        """
        Recursively find all JSON files under the specified path
        
        Args:
            search_path: Search path
            
        Returns:
            List of JSON file paths
        """
        search_path = Path(search_path)
        json_files = list(search_path.rglob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files")
        return json_files
    
    def parse_json_file(self, json_path: Path) -> Tuple[Dict, str]:
        """
        Parse JSON file to get segments information and video ID
        
        Args:
            json_path: JSON file path
            
        Returns:
            (JSON content, video ID)
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract video ID from filename
            # Example: 2SSipxwqy2A_aligned.json -> 2SSipxwqy2A
            video_id = json_path.stem.replace('filtered_', '')
            
            return data, video_id
        except Exception as e:
            logger.error(f"Failed to parse JSON file {json_path}: {e}")
            return None, None
    
    def get_video_path(self, json_path: Path, video_id: str) -> Path:
        """
        Infer video file path based on JSON path
        
        Args:
            json_path: JSON file path
            video_id: Video ID
            
        Returns:
            Video file path
        """
        # First, try to find the video file by searching in subdirectories
        video_filename = f"{video_id}.mp4"
        
        # Search for the video file recursively under the base path
        logger.info(f"Searching for video: {video_filename}")
        for video_file in self.video_base_path.rglob(video_filename):
            logger.info(f"Found video at: {video_file}")
            return video_file
        
        # If not found by searching, try to construct paths based on JSON structure
        # Extract relative path structure from JSON path
        # Example: .../filtered_output/20241103/02/2SSipxwqy2A_aligned.json
        # Video might be at: .../video/20241103/02/2SSipxwqy2A.mp4
        
        possible_paths = []
        
        # Try to extract date and subdirectory structure from JSON path
        try:
            # Get parts like ['20241103', '02'] from the JSON path
            parts = []
            for parent in json_path.parents:
                if parent.name and parent.name not in ['filtered_output', 'qwen_processing', 'whisper']:
                    parts.append(parent.name)
            parts.reverse()
            
            # Build possible paths
            if len(parts) >= 2:  # If we have date/subdirectory structure
                possible_paths.append(self.video_base_path / parts[0] / parts[1] / video_filename)
                possible_paths.append(self.video_base_path / parts[0] / video_filename)
            
        except Exception as e:
            logger.debug(f"Could not extract path structure: {e}")
        
        # Add other possible paths
        possible_paths.extend([
            self.video_base_path / video_filename,
            self.video_base_path / json_path.parent.name / video_filename,
        ])
        
        # Try each possible path
        for path in possible_paths:
            if path.exists():
                logger.info(f"Found video at constructed path: {path}")
                return path
        
        # If still not found, log all attempted paths
        logger.error(f"Video file not found: {video_filename}")
        logger.error(f"Searched recursively under: {self.video_base_path}")
        logger.error(f"Also tried these paths:")
        for path in possible_paths:
            logger.error(f"  - {path}")
        
        # Return the most likely path even if it doesn't exist
        return possible_paths[0] if possible_paths else self.video_base_path / video_filename
    
    def cut_video_segment(self, video_path: Path, start_time: float, end_time: float, 
                         output_path: Path) -> bool:
        """
        Cut video segment using ffmpeg
        
        Args:
            video_path: Input video path
            start_time: Start time (seconds)
            end_time: End time (seconds)
            output_path: Output video path
            
        Returns:
            Success status
        """
        # Extend time range
        extended_start = max(0, start_time - self.extend_seconds)
        extended_end = end_time + self.extend_seconds
        duration = extended_end - extended_start
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-ss', str(extended_start),
            '-t', str(duration),
            '-c', 'copy',  # Use copy codec for faster processing
            '-avoid_negative_ts', 'make_zero',
            '-y',  # Overwrite existing files
            str(output_path)
        ]
        
        try:
            logger.info(f"Cutting video: {video_path.name} [{extended_start:.1f}s - {extended_end:.1f}s]")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"ffmpeg error: {result.stderr}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Failed to cut video: {e}")
            return False
    
    def save_text(self, text: str, output_path: Path):
        """
        Save text content to file
        
        Args:
            text: Text content
            output_path: Output path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
    
    def process_json_file(self, json_path: Path):
        """
        Process a single JSON file
        
        Args:
            json_path: JSON file path
        """
        logger.info(f"Processing file: {json_path}")
        
        # Parse JSON
        data, video_id = self.parse_json_file(json_path)
        if not data or not video_id:
            return
        
        # Check segments
        segments = data.get('segments', [])
        if not segments:
            logger.info(f"No segments in file {json_path}, skipping")
            return
        
        # Get video path
        video_path = self.get_video_path(json_path, video_id)
        if not video_path.exists():
            if self.skip_missing:
                logger.warning(f"Video file does not exist: {video_path}, skipping this JSON file")
                return
            else:
                logger.error(f"Video file does not exist: {video_path}")
                return
        
        # Create output directory for each JSON file
        relative_path = json_path.relative_to(json_path.parents[3])
        output_dir = self.output_base_path / relative_path.parent / video_id
        
        # Process each segment
        for i, segment in enumerate(segments):
            start_time = segment['start']
            end_time = segment['end']
            text = segment.get('text', '')
            
            # Generate output filename
            segment_name = f"{video_id}_segment_{i+1:03d}_{int(start_time)}-{int(end_time)}"
            video_output = output_dir / f"{segment_name}.mp4"
            text_output = output_dir / f"{segment_name}.txt"
            
            # Cut video
            success = self.cut_video_segment(video_path, start_time, end_time, video_output)
            
            if success:
                # Save text
                self.save_text(text, text_output)
                logger.info(f"Successfully processed segment {i+1}/{len(segments)}")
                self._segments_processed += 1
            else:
                logger.error(f"Failed to process segment {i+1}/{len(segments)}")
    
    def process_all_files(self, search_path: str):
        """
        Process all JSON files under the specified path
        
        Args:
            search_path: Search path
        """
        json_files = self.find_json_files(search_path)
        
        total = len(json_files)
        processed = 0
        skipped = 0
        
        for i, json_path in enumerate(json_files, 1):
            logger.info(f"\nProgress: {i}/{total}")
            # Track if file was processed
            segments_before = getattr(self, '_segments_processed', 0)
            self.process_json_file(json_path)
            segments_after = getattr(self, '_segments_processed', 0)
            
            if segments_after > segments_before:
                processed += 1
            else:
                skipped += 1
        
        logger.info(f"\nProcessing complete!")
        logger.info(f"Total JSON files found: {total}")
        logger.info(f"Successfully processed: {processed}")
        logger.info(f"Skipped (no segments or video not found): {skipped}")


def main():
    parser = argparse.ArgumentParser(description='Cut video segments based on JSON files')
    parser.add_argument('search_path', help='JSON files search path')
    parser.add_argument('video_path', help='Video files base path')
    parser.add_argument('output_path', help='Output files path')
    parser.add_argument('--extend', type=int, default=5, help='Seconds to extend on each side (default: 5)')
    parser.add_argument('--skip-missing', action='store_true', help='Skip missing video files instead of stopping')
    
    args = parser.parse_args()
    
    # Create cutter instance
    cutter = VideoSegmentCutter(
        video_base_path=args.video_path,
        output_base_path=args.output_path,
        extend_seconds=args.extend,
        skip_missing=args.skip_missing
    )
    
    # Process all files
    cutter.process_all_files(args.search_path)


if __name__ == '__main__':
    main()