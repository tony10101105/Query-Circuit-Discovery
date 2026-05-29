import os
import sys
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm


def unzip_sae_data(base_dir: str):
    """
    For each folder matching *-resid-post-aa pattern, gunzip all .gz files
    under explanations/ and features/ subdirectories.
    
    Args:
        base_dir: Base directory containing the SAE data folders
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return
    
    # Find all folders matching the pattern
    # matching_folders = list(base_path.glob("*-resid-post-aa"))
    matching_folders = list(base_path.glob("*"))
    
    if not matching_folders:
        print(f"No folders matching '*' found in {base_dir}")
        return
    
    print(f"Found {len(matching_folders)} folders matching '*'")
    
    for folder in matching_folders:
        print(f"\nProcessing folder: {folder.name}")
        
        # Process explanations/ and features/ subdirectories
        for subdir_name in ["explanations", "features"]:
            subdir = folder / subdir_name
            
            if not subdir.exists():
                print(f"  Skipping {subdir_name}/ (not found)")
                continue
            
            # Find all .gz files in the subdirectory
            gz_files = list(subdir.glob("*.gz"))
            
            if not gz_files:
                print(f"  No .gz files found in {subdir_name}/")
                continue
            
            print(f"  Found {len(gz_files)} .gz files in {subdir_name}/")
            
            # Unzip each file
            for gz_file in tqdm(gz_files, desc=f"  Unzipping {subdir_name}"):
                output_file = gz_file.with_suffix('')  # Remove .gz extension
                
                # Skip if already unzipped
                if output_file.exists():
                    print(f"    Skipping {gz_file.name} (already unzipped)")
                    continue
                
                try:
                    with gzip.open(gz_file, 'rb') as f_in:
                        with open(output_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Optionally remove the .gz file after successful extraction
                    gz_file.unlink()
                    
                except Exception as e:
                    print(f"    Error unzipping {gz_file.name}: {e}")
    
    print("\nUnzipping complete!")


if __name__ == "__main__":
    base_directory = "./gpt2-small/32k"
    unzip_sae_data(base_directory)