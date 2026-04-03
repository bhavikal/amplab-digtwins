"""
Task-based EEG Preprocessing for HBN Dataset (ds005515)

This script processes task-based EEG recordings from the Healthy Brain Network dataset
and prepares them for GAN-based analysis. It handles multiple cognitive tasks and 
compares characteristics across task types.

Tasks included:
- DiaryOfAWimpyKid
- ThePresent
- FunwithFractals
- DespicableMe
- contrastChangeDetection
- surroundSupp
- symbolSearch
"""

import argparse
import mne
import numpy as np
import pandas as pd
import os
import glob
import warnings
import json
from pathlib import Path
from typing import Dict, Tuple, List

warnings.filterwarnings("ignore")

class HBNTaskDataLoader:
    """Load and organize HBN task-based EEG data from .set files"""
    
    TASK_NAMES = [
        'DiaryOfAWimpyKid',
        'ThePresent',
        'FunwithFractals',
        'DespicableMe',
        'contrastChangeDetection',
        'surroundSupp',
        'symbolSearch'
    ]
    
    def __init__(self, raw_data_dir: str):
        self.raw_data_dir = raw_data_dir
        self.task_files = {}
        self._organize_files()
    
    def _organize_files(self):
        """Organize .set files by task"""
        for task in self.TASK_NAMES:
            pattern = os.path.join(self.raw_data_dir, "sub-*/eeg/*_task-*" + task + "*_eeg.set")
            files = glob.glob(pattern)
            self.task_files[task] = sorted(files)
            print(f"{task}: {len(files)} files")
    
    def load_task_data(self, task_name: str, max_files: int = None) -> List[mne.io.Raw]:
        """Load all .set files for a given task"""
        files = self.get_task_files(task_name, max_files=max_files)
        
        raw_list = []
        for fpath in files:
            try:
                # MNE can read .set files directly
                raw = mne.io.read_raw_eeglab(fpath, preload=True)
                raw_list.append(raw)
            except Exception as e:
                print(f"Error loading {os.path.basename(fpath)}: {e}")
        
        return raw_list

    def get_task_files(self, task_name: str, max_files: int = None) -> List[str]:
        """Return .set file paths for a task without loading them into memory."""
        if task_name not in self.task_files:
            raise ValueError(f"Unknown task: {task_name}")

        files = self.task_files[task_name]
        if max_files:
            files = files[:max_files]
        return files
    
    def get_task_metadata(self) -> pd.DataFrame:
        """Return summary of available tasks"""
        data = []
        for task, files in self.task_files.items():
            data.append({
                'task': task,
                'n_files': len(files),
                'file_list': files
            })
        return pd.DataFrame(data)


class TaskEEGPreprocessor:
    """Preprocess task-based EEG data"""
    
    def __init__(self, sfreq_target: float = 128.0, l_freq: float = 0.5, h_freq: float = 45.0):
        self.sfreq_target = sfreq_target
        self.l_freq = l_freq
        self.h_freq = h_freq
    
    def preprocess_raw(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply standard preprocessing pipeline"""
        # Filter
        raw.filter(self.l_freq, self.h_freq, fir_design='firwin')
        
        # Resample if needed
        if raw.info['sfreq'] != self.sfreq_target:
            raw.resample(self.sfreq_target)
        
        # Re-reference to average
        raw.set_eeg_reference('average', projection=False)
        
        return raw
    
    def extract_segments(self, raw: mne.io.Raw, segment_duration: float = 2.0) -> np.ndarray:
        """
        Extract fixed-length segments from raw data
        Returns array of shape (n_segments, n_channels, n_timepoints)
        """
        sfreq = raw.info['sfreq']
        segment_samples = int(sfreq * segment_duration)
        
        data = raw.get_data()  # (n_channels, n_times)
        n_channels, n_times = data.shape
        
        # Extract non-overlapping segments
        segments = []
        for start_idx in range(0, n_times - segment_samples + 1, segment_samples):
            segment = data[:, start_idx:start_idx + segment_samples]
            segments.append(segment)
        
        return np.array(segments)  # (n_segments, n_channels, n_timepoints)
    
    def normalize_segments(self, segments: np.ndarray) -> np.ndarray:
        """
        Z-score normalization per segment per channel
        """
        normalized = []
        for segment in segments:
            # Normalize each channel independently
            norm_seg = np.zeros_like(segment)
            for ch_idx in range(segment.shape[0]):
                mean = segment[ch_idx].mean()
                std = segment[ch_idx].std() + 1e-8
                norm_seg[ch_idx] = (segment[ch_idx] - mean) / std
            normalized.append(norm_seg)
        
        return np.array(normalized).astype(np.float32)
    
    def process_task_files(self, raw_list: List[mne.io.Raw], 
                          segment_duration: float = 2.0) -> Tuple[np.ndarray, Dict]:
        """
        Process a list of raw files for a task
        Returns normalized segments and metadata
        """
        if not raw_list:
            raise ValueError("raw_list is empty; nothing to process.")

        all_segments = []
        segment_metadata = []
        ch_names = None
        sfreq = None
        
        for file_idx, raw in enumerate(raw_list):
            # Preprocess
            raw_proc = self.preprocess_raw(raw.copy())
            
            if ch_names is None:
                ch_names = raw_proc.ch_names
                sfreq = raw_proc.info['sfreq']
            
            # Extract segments
            segments = self.extract_segments(raw_proc, segment_duration)
            if segments.size == 0:
                continue

            all_segments.append(segments)
            
            # Track metadata
            source_file = 'unknown'
            if getattr(raw_proc, 'filenames', None):
                source_file = str(raw_proc.filenames[0])

            subject = 'unknown'
            for part in Path(source_file).parts:
                if part.startswith('sub-'):
                    subject = part
                    break

            for seg_idx in range(len(segments)):
                segment_metadata.append({
                    'file_idx': file_idx,
                    'segment_idx': seg_idx,
                    'subject': subject,
                    'source_file': source_file
                })

        if not all_segments:
            raise ValueError("No valid segments extracted from input files.")

        # Concatenate and normalize
        all_segments = np.concatenate(all_segments, axis=0)
        normalized_segments = self.normalize_segments(all_segments)
        
        metadata = {
            'n_segments': len(normalized_segments),
            'n_channels': normalized_segments.shape[1],
            'n_timepoints': normalized_segments.shape[2],
            'sfreq': sfreq,
            'segment_duration': segment_duration,
            'ch_names': [str(name) for name in ch_names],
            'n_files': len(raw_list),
            'segment_metadata': segment_metadata,
            'l_freq': self.l_freq,
            'h_freq': self.h_freq
        }
        
        return normalized_segments, metadata


class TaskEEGDataManager:
    """Manage saving and loading of preprocessed task data"""
    
    def __init__(self, output_dir: str = './task_gan_data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_task_dir(self, task_name: str, overwrite: bool = False) -> Tuple[Path, Path]:
        """Create task output folders and optionally clear prior individual segment files."""
        task_dir = self.output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        segments_dir = task_dir / f'{task_name}_segments_individual'
        if overwrite and segments_dir.exists():
            for old_file in segments_dir.glob('*.npy'):
                old_file.unlink()
        segments_dir.mkdir(exist_ok=True)

        return task_dir, segments_dir

    def save_task_metadata(self, task_name: str, metadata: Dict):
        """Save metadata JSON for a task."""
        task_dir = self.output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = task_dir / f'{task_name}_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def save_task_segments(self, task_name: str, segments: np.ndarray, 
                          metadata: Dict, overwrite: bool = False) -> str:
        """Save preprocessed segments for a task"""
        task_dir, segments_dir = self.prepare_task_dir(task_name, overwrite=overwrite)
        
        # Save full array
        segments_file = task_dir / f'{task_name}_segments.npy'
        np.save(segments_file, segments)
        
        # Save metadata
        self.save_task_metadata(task_name, metadata)
        
        for idx, seg in enumerate(segments):
            seg_file = segments_dir / f'{task_name}_segment_{idx:05d}.npy'
            np.save(seg_file, seg)
        
        print(f"\nSaved {task_name}:")
        print(f"  Total segments: {len(segments)}")
        print(f"  Shape: {segments.shape}")
        print(f"  Location: {task_dir}")
        print(f"  Individual segments: {len(list(segments_dir.glob('*.npy')))}")
        
        return str(task_dir)
    
    def load_task_segments(self, task_name: str) -> Tuple[np.ndarray, Dict]:
        """Load preprocessed segments for a task"""
        task_dir = self.output_dir / task_name
        
        segments_file = task_dir / f'{task_name}_segments.npy'
        metadata_file = task_dir / f'{task_name}_metadata.json'
        
        segments = np.load(segments_file)
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        return segments, metadata
    
    def get_task_summary(self) -> pd.DataFrame:
        """Get summary of all available task data"""
        data = []
        for task_dir in self.output_dir.iterdir():
            if task_dir.is_dir():
                task_name = task_dir.name
                metadata_file = task_dir / f'{task_name}_metadata.json'
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        meta = json.load(f)
                    data.append({
                        'task': task_name,
                        'n_segments': meta['n_segments'],
                        'n_channels': meta['n_channels'],
                        'n_timepoints': meta['n_timepoints'],
                        'sfreq': meta['sfreq']
                    })
        return pd.DataFrame(data)


def process_all_hbn_tasks(raw_data_dir: str, output_dir: str = './task_gan_data',
                         segment_duration: float = 2.0, max_files_per_task: int = None,
                         streaming: bool = True,
                         write_combined_array: bool = False,
                         overwrite: bool = False) -> Dict:
    """
    Process all available HBN tasks
    """
    print("=" * 80)
    print("HBN Task-Based EEG Preprocessing Pipeline")
    print("=" * 80)
    
    # Load data
    loader = HBNTaskDataLoader(raw_data_dir)
    print("\nAvailable tasks:")
    print(loader.get_task_metadata())
    
    # Initialize preprocessor and data manager
    preprocessor = TaskEEGPreprocessor()
    data_manager = TaskEEGDataManager(output_dir)
    
    # Process each task
    results = {}
    for task_name in loader.TASK_NAMES:
        if len(loader.task_files[task_name]) == 0:
            print(f"\nSkipping {task_name}: no files found")
            continue
        
        print(f"\n{'=' * 80}")
        print(f"Processing {task_name}...")
        print(f"{'=' * 80}")
        
        try:
            # Load task files
            max_files = max_files_per_task if max_files_per_task else None
            if streaming:
                file_paths = loader.get_task_files(task_name, max_files=max_files)
                if len(file_paths) == 0:
                    print(f"Could not find any files for {task_name}")
                    continue

                task_dir, segments_dir = data_manager.prepare_task_dir(task_name, overwrite=overwrite)
                print(f"Streaming {len(file_paths)} files to {segments_dir}")

                ch_names = None
                sfreq = None
                all_segments = []
                total_segments = 0
                files_processed = 0
                files_failed = 0

                for file_idx, fpath in enumerate(file_paths, start=1):
                    try:
                        raw = mne.io.read_raw_eeglab(fpath, preload=True)
                        raw_proc = preprocessor.preprocess_raw(raw)

                        if ch_names is None:
                            ch_names = raw_proc.ch_names
                            sfreq = raw_proc.info['sfreq']

                        segments = preprocessor.extract_segments(raw_proc, segment_duration)
                        if segments.size == 0:
                            continue

                        segments = preprocessor.normalize_segments(segments)

                        for seg in segments:
                            seg_file = segments_dir / f'{task_name}_segment_{total_segments:07d}.npy'
                            np.save(seg_file, seg)
                            total_segments += 1

                        if write_combined_array:
                            all_segments.append(segments)

                        files_processed += 1
                        if file_idx % 25 == 0:
                            print(f"  Processed {file_idx}/{len(file_paths)} files | segments so far: {total_segments}")

                    except Exception as file_error:
                        files_failed += 1
                        print(f"Error processing {os.path.basename(fpath)}: {file_error}")

                if total_segments == 0:
                    print(f"No valid segments extracted for {task_name}")
                    continue

                metadata = {
                    'n_segments': int(total_segments),
                    'n_channels': int(ch_names and len(ch_names) or 0),
                    'n_timepoints': int(segment_duration * sfreq) if sfreq else 0,
                    'sfreq': sfreq,
                    'segment_duration': segment_duration,
                    'ch_names': [str(name) for name in (ch_names or [])],
                    'n_files': int(files_processed),
                    'n_input_files': int(len(file_paths)),
                    'n_failed_files': int(files_failed),
                    'l_freq': preprocessor.l_freq,
                    'h_freq': preprocessor.h_freq,
                    'streaming_mode': True,
                    'individual_segments_dir': f'{task_name}_segments_individual'
                }
                data_manager.save_task_metadata(task_name, metadata)

                if write_combined_array and all_segments:
                    combined = np.concatenate(all_segments, axis=0)
                    np.save(task_dir / f'{task_name}_segments.npy', combined)
                    print(f"  Wrote combined array: {combined.shape}")

                print(f"Saved {task_name}: {total_segments} streamed segments")
                results[task_name] = {
                    'n_segments': total_segments,
                    'output_dir': str(task_dir),
                    'streaming_mode': True
                }
            else:
                raw_list = loader.load_task_data(task_name, max_files=max_files)

                if len(raw_list) == 0:
                    print(f"Could not load any files for {task_name}")
                    continue

                print(f"Loaded {len(raw_list)} files")

                # Process
                segments, metadata = preprocessor.process_task_files(
                    raw_list,
                    segment_duration=segment_duration
                )

                # Save
                task_dir = data_manager.save_task_segments(
                    task_name, segments, metadata, overwrite=overwrite
                )

                results[task_name] = {
                    'n_segments': len(segments),
                    'shape': segments.shape,
                    'output_dir': task_dir,
                    'streaming_mode': False
                }
        
        except Exception as e:
            print(f"Error processing {task_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'=' * 80}")
    print("Processing Complete - Summary")
    print(f"{'=' * 80}")
    summary_df = data_manager.get_task_summary()
    print(summary_df)
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess HBN task-based EEG data.')
    parser.add_argument('raw_data_dir', nargs='?', default='./raw/ds005515',
                        help='Path to raw ds005515 directory.')
    parser.add_argument('output_dir', nargs='?', default='./task_gan_data',
                        help='Output directory for preprocessed segments.')
    parser.add_argument('--segment-duration', '--segment_duration', type=float, default=2.0,
                        help='Segment duration in seconds (default: 2.0).')
    parser.add_argument('--max-files-per-task', '--max_files_per_task', type=int, default=None,
                        help='Optional max files per task for quick tests.')
    parser.add_argument('--streaming', dest='streaming', action='store_true', default=True,
                        help='Stream files and save individual segments without holding full task arrays in RAM (default: enabled).')
    parser.add_argument('--no-streaming', dest='streaming', action='store_false',
                        help='Disable streaming mode and use legacy in-memory task concatenation.')
    parser.add_argument('--write-combined-array', '--write_combined_array', action='store_true',
                        help='Also write task-level combined *_segments.npy arrays (can be memory-heavy for full dataset).')
    parser.add_argument('--overwrite', action='store_true',
                        help='Remove existing per-task individual segment files before writing new outputs.')
    args = parser.parse_args()

    # Run pipeline
    results = process_all_hbn_tasks(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        segment_duration=args.segment_duration,
        max_files_per_task=args.max_files_per_task,
        streaming=args.streaming,
        write_combined_array=args.write_combined_array,
        overwrite=args.overwrite
    )
    
    print("\nPreprocessing Results:")
    for task, res in results.items():
        print(f"  {task}: {res['n_segments']} segments")
