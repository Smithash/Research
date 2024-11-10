from PIL import Image
import os

def convert_to_jpeg(input_folder):
    # Walk through all folders and subfolders
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # Check if file is an image
            if file.lower().endswith(('.png', '.bmp', '.tiff', '.webp')):
                try:
                    # Open image
                    image_path = os.path.join(root, file)
                    img = Image.open(image_path)
                    
                    # Convert to RGB if necessary (for PNG with transparency)
                    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                        img = img.convert('RGB')
                    
                    # Create new filename
                    jpeg_path = os.path.splitext(image_path)[0] + '.jpg'
                    
                    # Save as JPEG
                    img.save(jpeg_path, 'JPEG', quality=95)
                    print(f"Converted: {file} -> {os.path.basename(jpeg_path)}")
                    
                    # Optionally, remove original file
                    # os.remove(image_path)
                    
                except Exception as e:
                    print(f"Error converting {file}: {str(e)}")

# Use the function
folder_path = "/Users/ashleasmith/Desktop/Postgrad CS/Research Project/Final Results"  # Replace with your folder path
convert_to_jpeg(folder_path)