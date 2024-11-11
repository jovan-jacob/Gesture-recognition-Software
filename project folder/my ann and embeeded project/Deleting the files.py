import os

file_path = ['Forward_recognition_model.h5','MP_Data']

for i in file_path:
    if os.path.exists(i):
        os.remove(i)
        print("File deleted successfully.")
    else:
        print("File not found.")