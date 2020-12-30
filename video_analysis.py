# importing libraries
import cv2
import numpy as np

import json

# Read Dictionary from dataset
with open('dataset/annotation_dict.json') as f:
    annotation_dict = json.load(f)

def keystoint(x):
    return {int(k): v for k, v in x.items()}

with open('dataset/labels_dict.json') as f:
    labels_dict = json.load(f, object_hook=keystoint)

# Function to open video
def openVideo(videoPath):
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(videoPath)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video  file")

        # Read until video is completed
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == "__main__":
    path = "dataset/examples/0000001"
    a = np.load(path + ".npy", allow_pickle=True)
    print(a)
    print("Number of Frames:", len(a))
    print("Number of Joints:", len(a[0]))
    openVideo(path + ".mp4")