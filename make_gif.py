dir = "/home/wenhao/projects/safepo/output_images"

import imageio
import os
import numpy as np

images = []

subdirs = os.listdir(dir)


for subdir in subdirs:
    subdir_path = os.path.join(dir, subdir)
    if os.path.isdir(subdir_path):
        # get all files in the replay folder
        files = os.listdir(subdir_path)
        files = [f for f in files if f.endswith(".png") and not f.startswith("episode_0")]

        # get the number of files
        print("Number of files: ", len(files))

        for i in range(len(files)):
            # Load the image
            img = imageio.imread(subdir_path + "/episode_1_frame_" + str(i+1) + ".png")
            images.append(img)

    imageio.mimsave(subdir_path + '/episode.gif', images)
    
    
