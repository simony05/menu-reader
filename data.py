import os
import csv
from PIL import Image
import numpy as np
import pandas as pd
import random

path = r"C:\Users\Simon\Desktop\chinese_characters"

num_images = []
# Number of images in each character folder
#with open('num.csv', 'w', newline = '', encoding = 'utf-8') as file:
    #writer = csv.writer(file)
    #field = ["index", "count"]
    #writer.writerow(field)

    #for i, folder in enumerate(os.listdir(path)):
        #count = 0
        #folder_path = r"C:\Users\Simon\Desktop\chinese_characters" + "\\" + folder
        #for file in os.listdir(folder_path):
            #count += 1
        #writer.writerow([i, count])

counts = pd.read_csv("num.csv")
counts = counts.drop(columns = ["index"])
print(counts.head())

# Open CSV file
with open('data.csv', 'w', newline = '', encoding = 'utf-8') as file:
    writer = csv.writer(file)
    field = ["image", "label"]

    writer.writerow(field)

    # Add images for each character (folder for each character) into csv with label as character
    for count, folder in enumerate(os.listdir(path)): # Length 52835
        folder_path = r"C:\Users\Simon\Desktop\chinese_characters" + "\\" + folder
        if counts.iloc[count]["count"] > 5:
            rand_nums = []
            for i in range(8):
                rand_nums.append(random.randint(0, counts.iloc[count]["count"]))

            for i, file in enumerate(os.listdir(folder_path)):
                if i in rand_nums:
                    file_path = folder_path + "\\" + file
                    image = Image.open(file_path)
                    pixels = list(image.getdata())
                    pixels = np.array(pixels).reshape(28, 28)
                    writer.writerow([pixels, count])
        else:
            for file in os.listdir(folder_path):
                file_path = folder_path + "\\" + file
                image = Image.open(file_path)
                pixels = list(image.getdata())
                pixels = np.array(pixels).reshape(28, 28)
                writer.writerow([0, count])

# Create a CSV for the characters based on index number
#with open('classes.csv', 'w', newline = '', encoding = 'utf-8') as file:
    #writer = csv.writer(file)
    #field = ["index", "character"]

    #writer.writerow(field)

    #for count, folder in enumerate(os.listdir(path)):
        #writer.writerow([count, folder])