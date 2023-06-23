import os

def get_img_list(img_dir):
    img_files = sorted([img for img in os.listdir(img_dir)
                        if img.endswith(".png")]
                        , key=lambda name: int(name.strip(".png")))
    
    return img_files