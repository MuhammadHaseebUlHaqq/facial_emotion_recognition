import os
from datasets import load_dataset
from PIL import Image

def save_split(dataset_split, base_path):
    # Mapping Hugging Face label IDs to Emotion names
    # Usually: 0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral
    # But let's check the dataset features if needed. Assuming standard mapping.
    label_map = {
        0: 'angry',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'sad',
        5: 'surprise',
        6: 'neutral'
    }
    
    for i, item in enumerate(dataset_split):
        img = item['image']
        label_id = item['label']
        emotion = label_map[label_id]
        
        folder = os.path.join(base_path, emotion)
        os.makedirs(folder, exist_ok=True)
        
        path = os.path.join(folder, f"{i}.png")
        img.save(path)
        
        if i % 1000 == 0:
            print(f"Saved {i} images to {base_path}...")

def main():
    print("Downloading FER2013 dataset from Hugging Face...")
    # 'deadpool/fer2013' or 'mertcobanov/fer2013' or 'mteb/fer2013'
    # Let's try 'mertcobanov/fer2013' or simply 'fer2013' (sometimes requires manual download)
    # A public accessible one is often 'mteb/fer2013' or similar. 
    # Let's use 'imagefolder' with a known public repo if possible.
    try:
        ds = load_dataset("mteb/fer2013")
        
        train_path = os.path.join(os.path.dirname(__file__), "..", "train")
        test_path = os.path.join(os.path.dirname(__file__), "..", "test")
        
        print("Extracting train set...")
        save_split(ds['train'], train_path)
        
        print("Extracting test set...")
        save_split(ds['test'], test_path)
        
        print("Done! Dataset is ready.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")

if __name__ == "__main__":
    main()
