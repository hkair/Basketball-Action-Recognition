import cv2
from tqdm import tqdm
import numpy as np
from vidaug import augmentors as vidaug
import json


def augmentVideo(annotation_dict, labels_dict, data_dir='dataset/examples/', output_dir='dataset/augmented-examples/'):
    """
    Takes in annotation dictionary, labels dictionary, path to the directory of data, and the output directory
    :param annotation_dict: Dictionary of basketball action annotations
    :param labels_dict: Dictionary matching the encoding to a Basketball Action
    :param data_dir: Path to Basketball Action Video Data
    :param output_dir: Output path of augmented videos
    :return:
    """

    with open(annotation_dict) as f:
        annotation_dict = json.load(f)
        video_list = list(annotation_dict.items())

    with open(labels_dict) as f:
        labels_dict = json.load(f, object_hook=keystoint)

    # Let's first visualize the distribution of actions in the
    count_dict = dict()
    for key in annotation_dict:
        if labels_dict[annotation_dict[key]] in count_dict:
            count_dict[labels_dict[annotation_dict[key]]] += 1
        else:
            count_dict[labels_dict[annotation_dict[key]]] = 1

    sorted_dict = {k: v for k, v in sorted(count_dict.items(), key=lambda item: item[1])}
    # Augments videos that have less than 2000 examples
    filtered_actions = [k for k, v in sorted_dict.items() if v <= 2000]

    # Transforms
    # sometimes = lambda aug: vidaug.Sometimes(0.5, aug)  # Used to apply augmentor with 50% probability
    # video_augmentation = vidaug.Sequential([
    #     sometimes(vidaug.Salt()),
    #     sometimes(vidaug.Pepper()),
    # ], random_order=True)

    augmented_annotation = dict()
    pbar = tqdm(video_list)

    # loop through video_list
    # create new label for augmented
    i = 0
    for video_id, action in pbar:
        path = data_dir + video_id + ".mp4"
        if labels_dict[action] in filtered_actions:

            # Rotate 30 degrees
            augmented_annotation[video_id + "_rotate_30"] = action
            rotateVideo(path, output_dir, video_id, 30)
            # Rotate 330 degrees
            augmented_annotation[video_id + "_rotate_330"] = action
            rotateVideo(path, output_dir, video_id, 330)

            # Translation - Right 32
            augmented_annotation[video_id + "_translate_32_0"] = action
            translateVideo(path, output_dir, video_id, (32, 0))
            # Translation - Left 32
            augmented_annotation[video_id + "_translate_-32_0"] = action
            translateVideo(path, output_dir, video_id, (-32, 0))

        i += 1
        pbar.set_description('Percentage {} '.format(i/len(video_list)))

    # Add to augmented_annotations
    with open('dataset/augmented_annotation_dict.json', 'w') as fp:
        json.dump(augmented_annotation, fp)

def rotateVideo(path, output_dir, video_id, degree):
    video = cv2.VideoCapture(path)
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_ROTATE = cv2.VideoWriter(output_dir + video_id + "_rotate_" + str(degree) + ".mp4", fourcc, fps,
                                    (frame_width, frame_height))
    # open video and collect frames
    while (video.isOpened()):
        # read video
        success, frame = video.read()
        if not success:
            break

        rotation_matrix = cv2.getRotationMatrix2D((frame_width / 2, frame_height / 2), degree, 1)
        rotated_frame = cv2.warpAffine(frame, rotation_matrix, (frame_width, frame_height))
        out_ROTATE.write(rotated_frame)

    video.release()
    out_ROTATE.release()

def translateVideo(path, output_dir, video_id, translate=(0,0)):
    video = cv2.VideoCapture(path)
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_TRANSLATE = cv2.VideoWriter(output_dir + video_id + "_translate_" + str(translate[0]) +"_" + str(translate[1]) + ".mp4", fourcc, fps,
                                    (frame_width, frame_height))
    # open video and collect frames
    while (video.isOpened()):
        # read video
        success, frame = video.read()
        if not success:
            break

        M = np.float32([[1, 0, translate[0]], [0, 1, translate[1]]])
        dst = cv2.warpAffine(frame, M, (frame_width, frame_height))
        out_TRANSLATE.write(dst)

    video.release()
    out_TRANSLATE.release()

def keystoint(x):
    return {int(k): v for k, v in x.items()}

if __name__ == "__main__":
    annotation_dict = "dataset/annotation_dict.json"
    labels_dict = "dataset/labels_dict.json"

    augmentVideo(annotation_dict, labels_dict)