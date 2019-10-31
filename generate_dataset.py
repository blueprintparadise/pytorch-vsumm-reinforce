"""
    Generate Dataset

    1. Converting video to frames
    2. Extracting features
    3. Getting change points
    4. User Summary ( for evaluation )

"""
import os, sys
sys.path.append('../')
from utils.KTS.cpd_auto import cpd_auto
from tqdm import tqdm
import math
import cv2
import numpy as np
import h5py
from feature_extractor import YouTube8MFeatureExtractor
from PIL import Image
import argparse
parser = argparse.ArgumentParser("name")
###
parser.add_argument('--video-dir', default='', help='')
parser.add_argument('--output-dir', default='', help='')

class Generate_Dataset:
    def __init__(self, video_path, save_path,frame_dir,train_data):
        self.dataset = {}
        self.video_list = []
        self.video_path = ''
        self.frame_root_path = frame_dir
        self.h5_file = h5py.File(save_path, 'w')
        self.extractor = YouTube8MFeatureExtractor()
        self._set_video_list(video_path)
        self.train_data = train_data

    def _set_video_list(self, video_path):
        if os.path.isdir(video_path):
            self.video_path = video_path
            self.video_list = os.listdir(video_path)
            self.video_list.sort()
        else:
            self.video_path = ''
            self.video_list.append(video_path)

        for idx, file_name in enumerate(self.video_list):
            self.dataset['video_{}'.format(idx+1)] = {}
            self.h5_file.create_group('video_{}'.format(idx+1))


    def _extract_feature(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame = cv2.resize(frame, (224, 224))
        res_pool5 = self.extractor.extract_rgb_frame_features(frame)
        #res_pool5 = self.resnet(frame)
        frame_feat =res_pool5 #res_pool5.cpu().data.numpy().flatten()
        #print(frame_feat.shape)
        return frame_feat

    def _get_change_points(self, video_feat, n_frame, fps):
        n = n_frame / fps
        m = int(math.ceil(n/2.0))
        K = np.dot(video_feat, video_feat.T)
        change_points, _ = cpd_auto(K, m, 1)
        change_points = np.concatenate(([0], change_points, [n_frame-1]))

        temp_change_points = []
        for idx in range(len(change_points)-1):
            segment = [change_points[idx], change_points[idx+1]-1]
            if idx == len(change_points)-2:
                segment = [change_points[idx], change_points[idx+1]]

            temp_change_points.append(segment)
        change_points = np.array(list(temp_change_points))

        temp_n_frame_per_seg = []
        for change_points_idx in range(len(change_points)):
            n_frame = change_points[change_points_idx][1] - change_points[change_points_idx][0] +1
            temp_n_frame_per_seg.append(n_frame)
        n_frame_per_seg = np.array(list(temp_n_frame_per_seg))

        return change_points, n_frame_per_seg

    # TODO : save dataset
    def _save_dataset(self):
        pass

    def generate_dataset(self):
        for video_idx, video_filename in enumerate(tqdm(self.video_list)):
            video_path = video_filename
            if os.path.isdir(self.video_path):
                video_path = os.path.join(self.video_path, video_filename)

            video_basename = os.path.basename(video_path).split('.')[0]
            print(video_filename)
            if not os.path.exists(os.path.join(self.frame_root_path, video_basename)):
                os.mkdir(os.path.join(self.frame_root_path, video_basename))

            video_capture = cv2.VideoCapture(video_path)
            video_dir = video_path
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_list = []
            picks = []
            video_feat = None
            video_feat_for_train = None
            print(n_frames)
            for frame_idx in tqdm(range(n_frames)):
                success, frame = video_capture.read()
                if success:
                    frame_feat = self._extract_feature(frame)

                    if frame_idx % 15 == 0:
                        picks.append(frame_idx)

                        if video_feat_for_train is None:
                            video_feat_for_train = frame_feat
                        else:
                            video_feat_for_train = np.vstack((video_feat_for_train, frame_feat))

                    if video_feat is None:
                        video_feat = frame_feat
                    else:
                        video_feat = np.vstack((video_feat, frame_feat))
                    #train_dataset
                    if self.train_data:
                        img_filename = "{}.jpg".format(str(frame_idx).zfill(6))
                        if not os.path.exists(os.path.join(self.frame_root_path, video_basename, img_filename)):
                            cv2.imwrite(os.path.join(self.frame_root_path, video_basename, img_filename), frame)

                    

                else:
                    break

            video_capture.release()

            change_points, n_frame_per_seg = self._get_change_points(video_feat, n_frames, fps)

            # self.dataset['video_{}'.format(video_idx+1)]['frames'] = list(frame_list)
            # self.dataset['video_{}'.format(video_idx+1)]['features'] = list(video_feat)
            # self.dataset['video_{}'.format(video_idx+1)]['picks'] = np.array(list(picks))
            # self.dataset['video_{}'.format(video_idx+1)]['n_frames'] = n_frames
            # self.dataset['video_{}'.format(video_idx+1)]['fps'] = fps
            # self.dataset['video_{}'.format(video_idx+1)]['change_points'] = change_points
            # self.dataset['video_{}'.format(video_idx+1)]['n_frame_per_seg'] = n_frame_per_seg

            self.h5_file['video_{}'.format(video_idx+1)]['features'] = list(video_feat_for_train)
            self.h5_file['video_{}'.format(video_idx+1)]['picks'] = np.array(list(picks))
            self.h5_file['video_{}'.format(video_idx+1)]['n_frames'] = n_frames
            self.h5_file['video_{}'.format(video_idx+1)]['fps'] = fps
            self.h5_file['video_{}'.format(video_idx + 1)]['video_name'] = video_filename.split('.')[0]
            self.h5_file['video_{}'.format(video_idx+1)]['change_points'] = change_points
            self.h5_file['video_{}'.format(video_idx+1)]['n_frame_per_seg'] = n_frame_per_seg
            self.h5_file['video_{}'.format(video_idx + 1)]['video_dir'] = video_dir


if __name__ == "__main__":
    args = parser.parse_args()
    video = args.video_dir
    output = args.output_dir
    frame_dir="./frames"
    gen = Generate_Dataset(video,output,frame_dir)
    gen.generate_dataset()
    gen.h5_file.close()
