"""
Kaggle Dataset Downloader
Helper script to download recommended datasets from Kaggle
"""

import subprocess
import os
from pathlib import Path

# Recommended dataset for the project
TELCO_CHURN = {
    "name": "Telco Customer Churn",
    "kaggle_path": "blastchar/telco-customer-churn",
    "file": "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "description": "Perfect for binary classification - predicting customer churn"
}

def check_kaggle_setup():
    """Check if Kaggle CLI is set up"""
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ Kaggle CLI is installed")
            return True
        else:
            print("❌ Kaggle CLI not working properly")
            return False
    except FileNotFoundError:
        print("❌ Kaggle CLI not installed")
        return False

def setup_instructions():
    """Print setup instructions"""
    print("\n" + "=" * 60)
    print("KAGGLE SETUP INSTRUCTIONS")
    print("=" * 60)
    print("\n1. Install Kaggle CLI:")
    print("   pip install kaggle")
    print("\n2. Get your Kaggle API credentials:")
    print("   a. Go to: https://www.kaggle.com/account")
    print("   b. Scroll to 'API' section")
    print("   c. Click 'Create New API Token'")
    print("   d. This downloads kaggle.json")
    print("\n3. Place kaggle.json in the right location:")
    print("   Mac/Linux: ~/.kaggle/kaggle.json")
    print("   Windows: C:\\Users\\<YourUsername>\\.kaggle\\kaggle.json")
    print("\n4. Set permissions (Mac/Linux only):")
    print("   chmod 600 ~/.kaggle/kaggle.json")
    print("\n" + "=" * 60)

def download_telco_churn():
    """Download the recommended Telco Churn dataset"""
    print("\n" + "=" * 60)
    print("DOWNLOADING TELCO CUSTOMER CHURN DATASET")
    print("=" * 60)
    
    # Create data/raw directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Downloading to: {data_dir}")
    print(f"📊 Dataset: {TELCO_CHURN['name']}")
    
    try:
        # Download dataset
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", TELCO_CHURN["kaggle_path"], "-p", str(data_dir)],
            check=True
        )
        
        print("\n✅ Download complete!")
        
        # Unzip if needed
        zip_file = data_dir / f"{TELCO_CHURN['kaggle_path'].split('/')[-1]}.zip"
        if zip_file.exists():
            print(f"\n📦 Unzipping...")
            subprocess.run(["unzip", "-o", str(zip_file), "-d", str(data_dir)], check=True)
            print("✅ Unzip complete!")
            
            # Remove zip file
            zip_file.unlink()
            print("🗑️  Removed zip file")
        
        # Rename file to dataset.csv for consistency
        source_file = data_dir / TELCO_CHURN["file"]
        target_file = data_dir / "dataset.csv"
        
        if source_file.exists():
            source_file.rename(target_file)
            print(f"\n✅ Dataset ready at: {target_file}")
            print("\n🎉 SUCCESS! You can now run: python dataset_selector.py data/raw/dataset.csv")
            return True
        else:
            print(f"\n⚠️  Expected file not found: {source_file}")
            print(f"   Files in {data_dir}:")
            for f in data_dir.iterdir():
                print(f"   - {f.name}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Download failed: {str(e)}")
        print("\n💡 Make sure you have:")
        print("   1. Kaggle account")
        print("   2. Kaggle CLI installed (pip install kaggle)")
        print("   3. API token configured (~/.kaggle/kaggle.json)")
        return False

def manual_download_instructions():
    """Instructions for manual download"""
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD (if Kaggle CLI doesn't work)")
    print("=" * 60)
    print("\n1. Go to: https://www.kaggle.com/blastchar/telco-customer-churn")
    print("2. Click 'Download' button (you may need to sign in)")
    print("3. Save the CSV file as: data/raw/dataset.csv")
    print("4. Run: python dataset_selector.py data/raw/dataset.csv")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    print("🚀 Kaggle Dataset Download Helper")
    
    if not check_kaggle_setup():
        setup_instructions()
        manual_download_instructions()
    else:
        print("\nReady to download!")
        response = input("\nDownload Telco Customer Churn dataset? (y/n): ")
        
        if response.lower() == 'y':
            success = download_telco_churn()
            if not success:
                manual_download_instructions()
        else:
            print("\n💡 You can download manually:")
            manual_download_instructions()
