import argparse
import os
from os.path import exists
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.plots import plot_one_box
from utils.torch_utils import time_synchronized, TracedModel
import time 
from tracker.byte_tracker import BYTETracker
from utils.visualize import plot_tracking
from tracking_utils.timer import Timer
from inference import Detect

ap = argparse.ArgumentParser()
DEFAULT_VIDEOPATH = "/content/drive/MyDrive/test/03_60fps.MOV"
DEFAULT_SAVEPATH = "image"
DEFAULT_FPS=10
ap.add_argument('-videopath', type=str, default=DEFAULT_VIDEOPATH,
                    help='set file')
ap.add_argument('-imagefile', type=str, default=DEFAULT_SAVEPATH,
                    help='set file')
ap.add_argument('-fps', type=int, default=DEFAULT_FPS,
                    help='set fps')
args = ap.parse_args()

DEFAULT_VIDEOPATH = args.videopath
DEFAULT_SAVEPATH = args.imagefile
DEFAULT_FPS=args.fps

print("DEFAULT_VIDEOPATH=",DEFAULT_VIDEOPATH)
print("DEFAULT_SAVEPATH=",DEFAULT_SAVEPATH)
print("DEFAULT_FPS=",DEFAULT_FPS)

def track_demo():
    video_path = DEFAULT_VIDEOPATH 
    txt_dir = "result"
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    
    # Detected
    conf_thres = 0.25
    iou_thres = 0.25
    img_size = 640
    weights = "weights/yolov7-tiny.pt"
    device = 0
    half_precision = True
    deteted = Detect(weights, device, img_size, conf_thres, iou_thres, single_cls=False, half_precision=half_precision, trace= False)
   
    # Tracking
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.8
    frame_rate = DEFAULT_FPS
    aspect_ratio_thresh = 1.6
    min_box_area = 10
    mot20_check = False
    res_file = os.path.join(txt_dir, video_path.split('/')[-1].replace('MOV', 'txt'))

    print(track_thresh, track_buffer, match_thresh, mot20_check, frame_rate)
    tracker = BYTETracker(track_thresh, track_buffer, match_thresh, mot20_check, frame_rate)
    timer = Timer()
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    results = []
    frame_index = 0
    while True:
        _,im0 = cap.read()
        frame_id += 1
        if _:
            height, width, _ = im0.shape
            t1 = time.time()
            dets = deteted.detecte(im0)
            online_targets = tracker.update(np.array(dets), [height, width], (height, width))
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save result for evaluation
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]},{tlwh[1]},{tlwh[2]},{tlwh[3]},{t.score:.2f},-1,-1,-1\n"
                    )
                    x = int(tlwh[0])
                    y = int(tlwh[1])
                    w = int(tlwh[2])
                    h = int(tlwh[3])
                    if x<0:x=0
                    elif y<0:y=0
                    elif w<0:w=0
                    elif h<0:h=0
                    corp_img=im0[y:y+h,x:x+w]
                    file=f"/content/drive/MyDrive/test/{DEFAULT_SAVEPATH}/{tid}"
                    if os.path.exists(file):
                      time_point = frame_index / frame_id
                      jpg_file=f"/content/drive/MyDrive/test/{DEFAULT_SAVEPATH}/{tid}/{tid}_{frame_id}_{time_point}.jpg"
                      cv2.imwrite(jpg_file, corp_img)
                      print(f"save{tid}successed")   
                      print(t1)                  
                    else:
                      time_point = frame_index / frame_id
                      os.system(f"mkdir /content/drive/MyDrive/test/{DEFAULT_SAVEPATH}/{tid}")
                      print(f"make {tid} succeed")
                      jpg_file=f"/content/drive/MyDrive/test/{DEFAULT_SAVEPATH}/{tid}/{tid}_{frame_id}_{time_point}.jpg"
                      cv2.imwrite(jpg_file, corp_img)
                      print(f"save{tid}successed")
                      print(t1)
                    # crop_image = im0[tlwh[0]:tlwh[]]
                    # print()  
            t2 = time.time()
            print(f"FPS:{1 /(t2-t1):.2f}")
            timer.toc()
            #print(1. / timer.average_time)
            online_im = plot_tracking(im0, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / 1 /(t2-t1))
            with open(res_file, 'w') as f:
                f.writelines(results)
            # cv2.imwrite(jpg_file, im0)
            # cv2.imshow("Frame", online_im)
        #     ch = cv2.waitKey(1)  # 1 millisecond
        #     if ch == ord("q") : 
        #         break
        else:
            break
if __name__ == "__main__":
    track_demo()
