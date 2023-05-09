#!/usr/bin/python
import matplotlib.pyplot as plt
import os
import random
import cv2

plt.figure(figsize=(4,4))
all_classes_names = os.listdir('mp_dataset')
random_range = random.sample(range(len(all_classes_names)), 5)
for counter, random_index in enumerate(random_range, 1):
    selected_class_Name = all_classes_names[random_index]
    img_files_names_list = os.listdir(f'mp_dataset/{selected_class_Name}')
    selected_img_file_name = random.choice(img_files_names_list)
    bgr_frame = cv2.imread(f'mp_dataset/{selected_class_Name}/{selected_img_file_name}')
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    plt.subplot(5, 4, counter)
    plt.subplots_adjust(hspace=0.5)
    plt.imshow(rgb_frame)
    plt.axis('off')
    plt.title(selected_class_Name)
plt.show()