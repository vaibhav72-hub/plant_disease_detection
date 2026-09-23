"""
PlantVillage Multi-Crop Dataset Downloader & Organizer
Downloads and prepares the 38-class PlantVillage dataset for training.
"""

import os
import sys
import shutil
import argparse

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

TARGET_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "PlantVillage")


def check_dataset_status(dataset_path):
    """Inspect and report on dataset directory structure and class counts."""
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset folder does not exist at: {dataset_path}")
        return False, 0, {}

    subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    if not subdirs:
        print(f"⚠️ Dataset folder exists at {dataset_path}, but contains no subdirectories.")
        return False, 0, {}

    class_counts = {}
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    total_images = 0

    for sub in sorted(subdirs):
        sub_path = os.path.join(dataset_path, sub)
        count = sum(1 for f in os.listdir(sub_path) if os.path.splitext(f.lower())[1] in valid_exts)
        class_counts[sub] = count
        total_images += count

    print("\n" + "=" * 60)
    print(f"🌿 PlantVillage Dataset Summary: {len(class_counts)} Classes Found")
    print(f"📁 Path: {dataset_path}")
    print(f"🖼️ Total Images: {total_images:,}")
    print("=" * 60)

    # Show first 10 classes and last few as sample
    sample_classes = list(class_counts.items())
    for name, cnt in sample_classes[:10]:
        print(f"  • {name:<45} : {cnt:>5} images")
    if len(sample_classes) > 10:
        print(f"  ... and {len(sample_classes) - 10} more classes.")

    print("=" * 60)
    return True, total_images, class_counts


def find_plantvillage_root(search_dir):
    """Find the directory that directly contains class folders like Tomato___Early_blight."""
    for root, dirs, files in os.walk(search_dir):
        # Look for characteristic PlantVillage folder naming
        indicators = [d for d in dirs if "Tomato___" in d or "Potato___" in d or "Apple___" in d]
        if len(indicators) >= 3:
            return root
    return None


def download_dataset():
    """Download PlantVillage dataset using kagglehub and set up directory."""
    print("🚀 Starting PlantVillage Multi-Crop Dataset Download...")
    print("This will download the complete multi-plant disease dataset (~1.5 GB).")
    
    try:
        import kagglehub
    except ImportError:
        print("❌ 'kagglehub' package not found. Installing now...")
        os.system(f'"{sys.executable}" -m pip install kagglehub')
        import kagglehub

    print("\n📥 Fetching 'emmarex/plantdisease' dataset via kagglehub...")
    print("(Kagglehub caches downloaded datasets locally and requires no manual extraction)")
    
    try:
        download_path = kagglehub.dataset_download("emmarex/plantdisease")
        print(f"✅ Download complete! Cached at: {download_path}")
    except Exception as e:
        print(f"❌ Failed to download via kagglehub: {e}")
        print("\nAlternative options:")
        print("1. Download 'PlantVillage' dataset directly from Kaggle: https://www.kaggle.com/datasets/emmarex/plantdisease")
        print(f"2. Extract the folders directly into: {TARGET_DATASET_DIR}")
        return False

    source_dir = find_plantvillage_root(download_path)
    if not source_dir:
        print(f"⚠️ Could not automatically locate PlantVillage class folders inside {download_path}")
        print(f"Please inspect {download_path} and move the class folders into {TARGET_DATASET_DIR}")
        return False

    print(f"\n📂 Found dataset classes at: {source_dir}")
    os.makedirs(os.path.dirname(TARGET_DATASET_DIR), exist_ok=True)

    # If TARGET_DATASET_DIR exists and has contents, don't re-copy
    if os.path.exists(TARGET_DATASET_DIR) and len(os.listdir(TARGET_DATASET_DIR)) > 0:
        print(f"ℹ️ Target dataset folder already exists at: {TARGET_DATASET_DIR}")
    else:
        print(f"📦 Setting up dataset in: {TARGET_DATASET_DIR}")
        # Try junction/symlink on Windows, otherwise copy
        linked = False
        try:
            if not os.path.exists(TARGET_DATASET_DIR):
                os.symlink(source_dir, TARGET_DATASET_DIR, target_is_directory=True)
                print("🔗 Created symbolic link to cached dataset.")
                linked = True
        except Exception:
            linked = False

        if not linked:
            print("📋 Copying dataset files to local project directory (this may take 1-2 minutes)...")
            if os.path.exists(TARGET_DATASET_DIR):
                shutil.rmtree(TARGET_DATASET_DIR)
            shutil.copytree(source_dir, TARGET_DATASET_DIR)
            print("✅ Dataset successfully copied!")

    # Check status
    ok, count, _ = check_dataset_status(TARGET_DATASET_DIR)
    if ok and count > 0:
        print("\n🎉 Dataset setup complete and verified!")
        print("👉 You can now train the model by running:")
        print("   python train_model.py\n")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="PlantVillage Multi-Crop Dataset Utility")
    parser.add_argument("--check", action="store_true", help="Only check existing dataset without downloading")
    args = parser.parse_args()

    if args.check:
        ok, count, _ = check_dataset_status(TARGET_DATASET_DIR)
        if not ok or count == 0:
            sys.exit(1)
        return

    # Check if dataset already exists
    if os.path.exists(TARGET_DATASET_DIR):
        ok, count, _ = check_dataset_status(TARGET_DATASET_DIR)
        if ok and count > 100:
            print("\n✨ PlantVillage dataset is already downloaded and ready to use!")
            print("👉 Run 'python train_model.py' to train on this dataset.")
            return

    download_dataset()


if __name__ == "__main__":
    main()
