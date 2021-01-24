# importing libraries
import cv2
import numpy as np

import json

# Read Dictionary from dataset
with open('../dataset/annotation_dict.json') as f:
    annotation_dict = json.load(f)

def keystoint(x):
    return {int(k): v for k, v in x.items()}

with open('../dataset/labels_dict.json') as f:
    labels_dict = json.load(f, object_hook=keystoint)

# Function to open and play video
def playVideo(videoPath, pose=False):
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(videoPath + ".mp4")
    joints = np.load(videoPath + ".npy", allow_pickle=True)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video  file")

    # Read until video is completed
    i = 0
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Display the resulting frame
            cv2.imshow('Frame', frame)
            if pose:
                draw_joints(frame, joints[i])
            i += 1

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

def draw_joints(image, joints):
    joint_dict = {0: "Nose", 1: "Neck", 2: "Right Shoulder", 3: "ElbowRight", 4: "WristRight", 5: "LeftShoulder",
                  6: "ElbowLeft", 7: "WristLeft", 8: "Hip Right",
                  9: "RightKnee", 10: "Ankle Right", 11: "LeftHip", 12: "Knee Left", 13: "Ankle Left"}
    for key, val in joints.items():
        joint = cv2.circle(image, val, radius=0, color=(0, 0, 255), thickness=-3)
        text_loc = (val[0] + 10, val[1] + 10)
        if (key <= 13):
            cv2.putText(image, joint_dict[key], text_loc, cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)
        cv2.imshow('Frame', joint)

def showActions(videoPath, action_labels, nums):
    # show videos of given action
    for i in range(0, nums):
        a = np.load(videoPath + action_labels[i][0] + ".npy", allow_pickle=True)
        print(a)
        print("Number of Frames:", len(a))
        print("Number of Joints:", len(a[0]))
        print("Action:", action_labels[i][1])
        playVideo(videoPath + action_labels[i][0], pose=True)

if __name__ == "__main__":
    path = "../dataset/examples/"

    with open("../dataset/annotation_dict.json") as f:
        annotation_dict = list(json.load(f).items())

    # Defence - "block"
    stop = [val for val in annotation_dict if val[1] == 0]
    # Passing
    ball_pass = [val for val in annotation_dict if val[1] == 1]
    # Race or Running
    race = [val for val in annotation_dict if val[1] == 2]
    # Dribble
    pallegio = [val for val in annotation_dict if val[1] == 3]
    # Shooting
    shooting = [val for val in annotation_dict if val[1] == 4]
    # Ball In Hand
    ballinhand = [val for val in annotation_dict if val[1] == 5]
    # Defensive Position
    defence = [val for val in annotation_dict if val[1] == 6]
    # Pick Attempt
    pick_attempt = [val for val in annotation_dict if val[1] == 7]
    # No Action - Just standing
    noaction = [val for val in annotation_dict if val[1] == 8]
    # walk
    walk = [val for val in annotation_dict if val[1] == 9]

    # Show 10 videos
    showActions(path, walk, 10)