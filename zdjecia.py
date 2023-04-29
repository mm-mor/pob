import os
from PIL import Image


input_paths = ["laminat", "gres", "sciana"]

output_paths = ["laminat1", "gres1", "sciana1"]

for input_path, output_path in zip(input_paths, output_paths):

    for filename in os.listdir(input_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):

            img = Image.open(os.path.join(input_path, filename))

            width, height = img.size
            for i in range(0, width, 128):
                for j in range(0, height, 128):
                    box = (i, j, i+128, j+128)
                    region = img.crop(box)

                    region.save(os.path.join(output_path, f"{filename}_{i}_{j}.png"))
