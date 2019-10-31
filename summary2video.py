import h5py
import cv2
import os
import os.path as osp
import numpy as np
import argparse
dict1 = {
'video_1':'Air_Force_One',
'video_10' :'Excavators river crossing',
'video_11' :'Fire Domino',
'video_12' :'Jumps',
'video_13' :'Kids_playing_in_leaves',
'video_14' :'Notre_Dame',
'video_15' :'Paintball',
'video_16' :'Playing_on_water_slide',
'video_17' :'Saving dolphines',
'video_18' :'Scuba',
'video_19' :'St Maarten Landing',
'video_2' :'Base jumping/Base jumping',
'video_20' :'Statue of Liberty',
'video_21' :'Uncut_Evening_Flight',
'video_22' :'Valparaiso_Downhill',
'video_23' :'car_over_camera',
'video_24' :'paluma_jump',
'video_25' :'playing_ball',
'video_3' :'Bearpark_climbing',
'video_4' :'Bike Polo',
'video_5' :'Bus_in_Rock_Tunnel',
'video_6' :'Car_railcrossing',
'video_7' :'Cockpit_Landing',
'video_8' :'Cooking',
'video_9' :'Eiffel Tower'
}


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, required=True, help="path to h5 result file")
parser.add_argument('-d', '--frm-dir', type=str, required=True, help="path to frame directory")
parser.add_argument('-i', '--idx', type=int, default=3, help="which key to choose")
parser.add_argument('--fps', type=int, default=30, help="frames per second")
parser.add_argument('--width', type=int, default=640, help="frame width")
parser.add_argument('--height', type=int, default=480, help="frame height")
parser.add_argument('--save-dir', type=str, default='log', help="directory to save")
parser.add_argument('--save-name', type=str, default='summary.mp4', help="video name to save (ends with .mp4)")
args = parser.parse_args()

def frm2video(frm_dir, summary, vid_writer,key):
    for idx, val in enumerate(summary):
        if val == 1:
            # here frame name starts with '000001.jpg'
            # change according to your need
            frm_name = str(idx+1).zfill(5) + '.jpg'
            frm_path = osp.join(frm_dir,"train", frm_name)
            print(frm_path)
            frm = cv2.imread(frm_path)
            frm = cv2.resize(frm, (args.width, args.height))
            vid_writer.write(frm)
'''
dict1 = {
'video_1':'Air_Force_One'
'video_10' :'Excavators river crossing'
'video_11' :'Fire Domino'
'video_12' :'Jumps'
'video_13' :'Kids_playing_in_leaves'
'video_14' :'Notre_Dame'
'video_15' :'Paintball'
'video_16' :'Playing_on_water_slide'
'video_17' :'Saving dolphines'
'video_18' :'Scuba'
'video_19' :'St Maarten Landing'
'video_2' :#'Base jumping'
'video_20' :'Statue of Liberty'
'video_21' :'Uncut_Evening_Flight'
'video_22' :'Valparaiso_Downhill'
'video_23' :'car_over_camera'
'video_24' :'paluma_jump'
'video_25' :'playing_ball'
'video_3' :'Bearpark_climbing'
'video_4' :'Bike Polo'
'video_5' :'Bus_in_Rock_Tunnel'
'video_6' :'Car_railcrossing'
'video_7' :'Cockpit_Landing'
'video_8' :'Cooking'
'video_9' :'Eiffel Tower'
}
'''


if __name__ == '__main__':
    if not osp.exists(args.save_dir):
        os.mkdir(args.save_dir)

    h5_res = h5py.File(args.path, 'r')

    ####遍历生成###
    for idx1 in range(len(list(h5_res.keys()))):
        print(idx1)
        key = list(h5_res.keys())[idx1]
        print(list(h5_res.keys()))
        summary = h5_res[key]['machine_summary'][...]
        print(summary.shape)
        import h5py
        import sys
        print(osp.join(args.save_dir, "train", args.save_name))
        if not os.path.isdir(osp.join(args.save_dir, dict1[key])):
            os.mkdir(osp.join(args.save_dir, dict1[key]))

        if key == 'video_1':
            vid_writer = cv2.VideoWriter(
                #osp.join(args.save_dir, dict1[key], args.save_name),
                osp.join(args.save_dir,'train', args.save_name),
                cv2.VideoWriter_fourcc(*'MP4V'),
                args.fps,
                (args.width, args.height),
            )
            frm2video(args.frm_dir, summary, vid_writer,key)
            vid_writer.release()
    h5_res.close()

    '''
    h5_file_name = ("./datasets/eccv16_dataset_summe_google_pool5.h5")
    f = h5py.File(h5_file_name, 'r')
    for key in f.keys():
        print("{} : {}".format(key, f[key]['video_name'].value))    
    '''



