"""Task 1: Dataset exploration and loading functions."""

import shutil
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd


def download_dataset(use_local_cache=True):
    local_data_dir = Path("data")
    
    if use_local_cache and local_data_dir.exists():
        activity_folders = [d for d in local_data_dir.iterdir() 
                           if d.is_dir() and (d / 'Accelerometer.csv').exists()]
        if len(activity_folders) > 0:
            print(f"Using local cache in {local_data_dir.absolute()}")
            print(f"Found {len(activity_folders)} activity folders in local cache")
            return local_data_dir
    
    print("Downloading dataset from Kaggle...")
    kaggle_cache_path = kagglehub.dataset_download("edgeimpulse/activity-detection")
    kaggle_path = Path(kaggle_cache_path)
    
    print(f"Dataset downloaded to Kaggle cache: {kaggle_path}")
    
    if use_local_cache:
        print(f"Copying dataset to local cache: {local_data_dir.absolute()}")
        local_data_dir.mkdir(exist_ok=True)
        
        activity_folders = [d for d in kaggle_path.iterdir() 
                         if d.is_dir() and (d / 'Accelerometer.csv').exists()]
        
        for folder in activity_folders:
            dest_folder = local_data_dir / folder.name
            if not dest_folder.exists():
                print(f"  Copying {folder.name}...")
                shutil.copytree(folder, dest_folder)
            else:
                print(f"  Skipping {folder.name} (already exists)")
        
        print(f"Local cache ready at {local_data_dir.absolute()}")
        return local_data_dir
    else:
        return kaggle_path


def load_activity_data(dataset_path):
    data_list = []
    metadata = {
        'participants': set(),
        'labels': [],
        'days': {},
        'demographics': {}
    }
    
    print("Loading real data from activity folders...")
    
    activity_folders = []
    for item in dataset_path.iterdir():
        if item.is_dir():
            acc_file = item / 'Accelerometer.csv'
            if acc_file.exists():
                activity_folders.append(item)
    
    print(f"Found {len(activity_folders)} activity folders")
    
    for folder in activity_folders:
        try:
            folder_name = folder.name
            activity_label = folder_name.split('-')[0].lower()
            
            acc_file = folder / 'Accelerometer.csv'
            df = pd.read_csv(acc_file)
            
            gyro_file = folder / 'Gyroscope.csv'
            if gyro_file.exists():
                try:
                    gyro_df = pd.read_csv(gyro_file)
                    if 'time' in df.columns and 'time' in gyro_df.columns:
                        df = pd.merge(df, gyro_df, on='time', suffixes=('_acc', '_gyr'), how='outer')
                        df = df.sort_values('time').reset_index(drop=True)
                    elif 'seconds_elapsed' in df.columns and 'seconds_elapsed' in gyro_df.columns:
                        df = pd.merge(df, gyro_df, on='seconds_elapsed', suffixes=('_acc', '_gyr'), how='outer')
                        df = df.sort_values('seconds_elapsed').reset_index(drop=True)
                    else:
                        if len(df) == len(gyro_df):
                            for col in gyro_df.columns:
                                if col not in df.columns:
                                    df[col] = gyro_df[col].values
                except Exception as e:
                    print(f"Warning: Could not merge gyroscope data for {folder_name}: {e}")

            gps_file = folder / 'LocationGps.csv'
            if gps_file.exists():
                try:
                    gps_df = pd.read_csv(gps_file)
                    
                    merge_col = None
                    if 'seconds_elapsed' in df.columns and 'seconds_elapsed' in gps_df.columns:
                        merge_col = 'seconds_elapsed'
                    elif 'time' in df.columns and 'time' in gps_df.columns:
                        merge_col = 'time'
                    
                    if merge_col is not None and 'speed' in gps_df.columns:
                        gps_subset = gps_df[[merge_col, 'speed']].copy()
                        df = pd.merge(df, gps_subset, on=merge_col, how='left')
                except Exception as e:
                    print(f"Warning: Could not merge GPS data for {folder_name}: {e}")
            
            df['activity'] = activity_label
            participant_id = folder_name
            
            data_list.append({
                'participant': participant_id,
                'file': acc_file.name,
                'dataframe': df
            })
            
            metadata['participants'].add(participant_id)
            metadata['labels'].append(activity_label)
            
        except Exception as e:
            print(f"Error loading {folder.name}: {e}")
            continue
    
    print(f"Successfully loaded {len(data_list)} activity sessions")
    print(f"Activity types found: {set(metadata['labels'])}")
    
    return data_list, metadata


def create_dataset_summary(data_list, metadata):
    n_participants = len(metadata['participants'])
    
    participant_files = {}
    for item in data_list:
        p_id = item['participant']
        if p_id not in participant_files:
            participant_files[p_id] = 0
        participant_files[p_id] += 1
    
    avg_files_per_participant = np.mean(list(participant_files.values()))
    
    unique_labels = set(metadata['labels'])
    total_labels = len(metadata['labels'])
    
    summary = pd.DataFrame({
        'Metric': [
            'Number of Participants',
            'Avg Files per Participant',
            'Total Labels/Activities',
            'Unique Activity Types',
            'Total Data Files'
        ],
        'Value': [
            n_participants,
            f"{avg_files_per_participant:.1f}",
            total_labels,
            len(unique_labels),
            len(data_list)
        ]
    })
    
    return summary, unique_labels

