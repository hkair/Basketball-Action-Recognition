# Util.py
import json
import glob
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class BasketballDataset(Dataset):
    """SpaceJam: a Dataset for Basketball Action Recognition."""

    def __init__(self, annotation_dict, label_dict, video_dir="dataset/examples/", transform=None, poseData=False):
        with open(annotation_dict) as f:
            self.video_list = list(json.load(f).items())
        with open(label_dict) as l:
            self.label_dict = json.load(l, object_hook=self.keystoint)

        self.video_dir = video_dir
        self.poseData = poseData
        self.transform = transform

    def __len__(self):
        # return length of none-flipped videos in directory
        return len(self.video_list)

    def __getitem__(self, idx):
        video_id = self.video_list[idx][0]

        # Transform Video to Tensor
        if self.transform:
            video = self.transform(self.video_dir+video_id+".mp4")

        encoding = np.squeeze(np.eye(10)[np.array([0,1,2,3,4,5,6,7,8,9]).reshape(-1)])
        if self.poseData:
            joints = np.load(self.video_dir + video_id + ".npy", allow_pickle=True)
            sample = {'video_id': video_id, 'joints': joints, 'action': torch.from_numpy(np.array(encoding[self.video_list[idx][1]])),'class': self.video_list[idx][1]}
        else:
            sample = {'video_id': video_id, 'video': video, 'action': torch.from_numpy(np.array(encoding[self.video_list[idx][1]])), 'class': self.video_list[idx][1]}

        return sample

    def keystoint(self, x):
        return {int(k): v for k, v in x.items()}


class VideoFilePathToTensor(object):
    """ load video at given file path to torch.Tensor (C x L x H x W, C = 3)
        It can be composed with torchvision.transforms.Compose().

    Args:
        max_len (int): Maximum output time depth (L <= max_len). Default is None.
            If it is set to None, it will output all frames.
        fps (int): sample frame per seconds. It must lower than or equal the origin video fps.
            Default is None.
        padding_mode (str): Type of padding. Default to None. Only available when max_len is not None.
            - None: won't padding, video length is variable.
            - 'zero': padding the rest empty frames to zeros.
            - 'last': padding the rest empty frames to the last frame.
    """

    def __init__(self, max_len=None, fps=None, padding_mode=None):
        self.max_len = max_len
        self.fps = fps
        assert padding_mode in (None, 'zero', 'last')
        self.padding_mode = padding_mode
        self.channels = 3  # only available to read 3 channels video

    def __call__(self, path):
        """
        Args:
            path (str): path of video file.

        Returns:
            torch.Tensor: Video Tensor (C x L x H x W)
        """

        # open video file
        cap = cv2.VideoCapture(path)
        assert (cap.isOpened())

        # calculate sample_factor to reset fps
        sample_factor = 1
        if self.fps:
            old_fps = cap.get(cv2.CAP_PROP_FPS)  # fps of video
            sample_factor = int(old_fps / self.fps)
            assert (sample_factor >= 1)

        # init empty output frames (C x L x H x W)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        time_len = None
        if self.max_len:
            # time length has upper bound
            if self.padding_mode:
                # padding all video to the same time length
                time_len = self.max_len
            else:
                # video have variable time length
                time_len = min(int(num_frames / sample_factor), self.max_len)
        else:
            # time length is unlimited
            time_len = int(num_frames / sample_factor)

        frames = torch.FloatTensor(self.channels, time_len, height, width)

        for index in range(time_len):
            frame_index = sample_factor * index

            # read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                # successfully read frame
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                # (H x W x C) to (C x H x W)
                frame = frame.permute(2, 0, 1)
                frames[:, index, :, :] = frame.float()
            else:
                # reach the end of the video
                if self.padding_mode == 'zero':
                    # fill the rest frames with 0.0
                    frames[:, index:, :, :] = 0
                elif self.padding_mode == 'last':
                    # fill the rest frames with the last frame
                    assert (index > 0)
                    frames[:, index:, :, :] = frames[:, index - 1, :, :].view(self.channels, 1, height, width)
                break

        frames /= 255
        cap.release()
        return frames

class BasketballDatasetTensor(Dataset):
    """SpaceJam: a Dataset for Basketball Action Recognition."""

    def __init__(self, annotation_dict, video_dir="dataset/examples/", data_dir="tensor-dataset/", transform=None, poseData=False):
        with open(annotation_dict) as f:
            self.video_list = list(json.load(f).items())

        self.video_dir = video_dir
        self.data_dir = data_dir
        self.poseData = poseData
        self.transform = transform

    def __len__(self):
        # return length of none-flipped videos in directory
        return len(self.video_list)

    def __getitem__(self, idx):
        video_id = self.video_list[idx][0]

        encoding = np.squeeze(np.eye(10)[np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1)])
        #print(encoding)
        if self.poseData:
            joints = np.load(self.video_dir + video_id + ".npy", allow_pickle=True)
            sample = {'video_id': video_id, 'joints': joints, 'action': torch.from_numpy(np.array(encoding[self.video_list[idx][1]])), 'class': self.video_list[idx][1]}
        else:
            video = torch.load(self.data_dir + video_id + ".pt")
            sample = {'video_id': video_id, 'video': video, 'action': torch.from_numpy(np.array(encoding[self.video_list[idx][1]])), 'class': self.video_list[idx][1]}

        return sample

def VideoToTensor(video_id, data_dir="dataset/examples/", output_dir="tensor-dataset/", max_len=None, fps=None, padding_mode=None):
    # open video file
    path = data_dir + video_id + ".mp4"
    cap = cv2.VideoCapture(path)
    assert (cap.isOpened())

    channels = 3

    # calculate sample_factor to reset fps
    sample_factor = 1
    if fps:
        old_fps = cap.get(cv2.CAP_PROP_FPS)  # fps of video
        sample_factor = int(old_fps / fps)
        assert (sample_factor >= 1)

    # init empty output frames (C x L x H x W)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    time_len = None
    if max_len:
        # time length has upper bound
        if padding_mode:
            # padding all video to the same time length
            time_len = max_len
        else:
            # video have variable time length
            time_len = min(int(num_frames / sample_factor), max_len)
    else:
        # time length is unlimited
        time_len = int(num_frames / sample_factor)

    frames = torch.FloatTensor(channels, time_len, height, width)

    for index in range(time_len):
        frame_index = sample_factor * index

        # read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            # successfully read frame
            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames[:, index, :, :] = frame.float()
        else:
            # reach the end of the video
            if padding_mode == 'zero':
                # fill the rest frames with 0.0
                frames[:, index:, :, :] = 0
            elif padding_mode == 'last':
                # fill the rest frames with the last frame
                assert (index > 0)
                frames[:, index:, :, :] = frames[:, index - 1, :, :].view(channels, 1, height, width)
            break

    frames /= 255
    cap.release()
    torch.save(frames, output_dir + video_id + ".pt")

def convertAllVideo(path="dataset/annotation_dict.json", data_dir="dataset/examples/", output_dir="tensor-dataset/"):
    # Let's convert all video to .pt files
    with open(path) as f:
        video_list = list(json.load(f).items())

    i = 1
    for video_id in video_list:
        print(video_id[0])
        print("Video: ", i)
        print(i/37085)
        VideoToTensor(video_id[0], data_dir, output_dir, max_len=16, fps=10, padding_mode='last')
        i += 1

def returnWeights(annotation_dict='dataset/annotation_dict.json', labels_dict='dataset/labels_dict.json'):
    # Read Dictionary from dataset
    with open(annotation_dict) as f:
        annotation_dict = json.load(f)

    def keystoint(x):
        return {int(k): v for k, v in x.items()}

    with open(labels_dict) as f:
        labels_dict = json.load(f, object_hook=keystoint)

    # Let's first visualize the distribution of actions in the
    count_dict = dict()
    for key in annotation_dict:
        if labels_dict[annotation_dict[key]] in count_dict:
            count_dict[labels_dict[annotation_dict[key]]] += 1
        else:
            count_dict[labels_dict[annotation_dict[key]]] = 1

    for key in count_dict:
        count_dict[key] = count_dict[key]/37085

    weights = []
    for key, val in labels_dict.items():
        if val != "discard":
            weights.append(count_dict[val])

    return weights

if __name__ == "__main__":
    # basketball_dataset = BasketballDataset(annotation_dict="dataset/annotation_dict.json",
    #                             label_dict="dataset/labels_dict.json",
    #                             transform=transforms.Compose([VideoFilePathToTensor(max_len=16, fps=10, padding_mode='last')]),
    #                             poseData=True)

    basketball_dataset = BasketballDatasetTensor(annotation_dict="dataset/annotation_dict.json",
                                                poseData=False)

    print(basketball_dataset[1]['action'])
    print(basketball_dataset[1]['class'])