# !/usr/bin/python3
# General libs
import numpy as np
import os
import time
import cv2
import torch
import shutil
import argparse
from sys import platform
from datetime import datetime
from models import *
from utils import *
from torch_utils import *
import pandas as pd
import  natsort
class Detect():
    def __init__(self, args_):
        self.opt = args_
        print(self.opt)
        img_size = (
        320, 192) if ONNX_EXPORT else self.opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
        out, source, weights, self.half, view_img, save_txt = self.opt.output, self.opt.source, self.opt.weights, self.opt.half, self.opt.view_img, self.opt.save_txt

        self.device = select_device(device='cpu' if ONNX_EXPORT else self.opt.device)
        self.model = Darknet(self.opt.cfg, img_size)
        self.model.load_state_dict(torch.load(weights, map_location=self.device)['model'])
        # Second-stage classifier
        self.classify = False
        if self.classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
            modelc.to(self.device).eval()
        # Eval mode
        self.model.to(self.device).eval()
        self.arry_size = 6
        # super().__init__('talker')
        # self._pub = self.create_publisher(Float32MultiArray, '/tl_bbox_info')
        # self._cv_bridge = CvBridge()
        # super().__init__('rectifier')
        # self._sub = self.create_subscription(Image, '/snowball/perception/traffic_light/cropped_roi', self.img_callback,
        #                                      10)
        print("ready to process rectifier----------------------------------------------------------")
        # input_folder="/home/mayank_s/Desktop/template/farm_2/farm_2_images2_scaled"
        # input_folder="/home/mayank_s/datasets/farminton/new_cropped_Demo/cfg_1llop"
        input_folder="/home/mayank_s/datasets/farminton/more_traffic_light_dataset_for_farmington/traffic_light"
        # input_folder="/home/mayank_s/datasets/farminton/more_traffic_light_dataset_for_farmington/0cc91a22-c8e2-4686-9428-b493e5b527ce"
        output_folder="/home/mayank_s/Desktop/yolo_output"
        self.output_path_crop='/home/mayank_s/Desktop/output_path_crop'
        self.bblabel=[]
        self.img_callback(input_folder,output_folder)
    # Camera image callback

    def img_callback(self, input_folder,output_folder):
        box_no=0
        # t = time.time()
        # bridge = cv_bridge.CvBridge()
        # cv_img = bridge.imgmsg_to_cv2(image_msg, 'passthrough')
        # print("image shape is ", cv_img.shape)
        # image_np = cv_img
        if not os.path.exists(input_folder):
            print("Input folder not found")
            return 1

        if not os.path.exists(output_folder):
            print("Output folder not present. Creating New folder...")
            os.makedirs(output_folder)
        for list_folder in os.listdir(input_folder):
            for root, a_file, filenames in os.walk(os.path.join(input_folder, list_folder)):
        #     for filenames in a_file:
                if (len(filenames) == 0):
                    print("Input folder is empty")
                    return 1
                time_start = time.time()
                filenames = natsort.natsorted(filenames, reverse=False)
                for filename in filenames:
                    try:
                        print("Creating object detection for file : {fn}".format(fn=filename), '\n')

                        file_path = (os.path.join(root, filename))
                        image_np = cv2.imread(file_path,1)

                        img,im0s = load_image_direct(image_np, img_size=416)
                        img = torch.from_numpy(img).to(self.device)
                        img = img.half() if self.half else img.float()  # uint8 to fp16/32
                        img /= 255.0  # 0 - 255 to 0.0 - 1.0
                        if img.ndimension() == 3:
                            img = img.unsqueeze(0)
                        # Inference
                        pred = self.model(img)[0].float() if self.half else self.model(img)[0]
                        # Apply NMS
                        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                                   agnostic=self.opt.agnostic_nms)

                        # Apply Classifier
                        if self.classify:
                            pred = apply_classifier(pred, modelc, img, im0s)
                        tmp = [-1.0, 10.0, 10.0, 10.0, 10.0, 10.0]
                        if pred[0] is not None:
                            # print('prediction before', pred[0])
                            # to select only traffic light
                            keep = torch.where(pred[0][:, 5] == 9)[0]
                            pred[0] = pred[0][keep]
                            # print('prediction after', pred[0])
                            # Process detections
                            for loop, det in enumerate(pred):  # detections per image
                                if det is not None and len(det):
                                    tmp = -np.ones(self.arry_size * int(len(det)) + 1)
                                    # Rescale boxes from img_size to im0 size
                                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                                    for i, boxes in enumerate(det):
                                        box_no=box_no+1
                                        c1, c2 = (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3]))
                                        # print(c2[0],c2[1])
                                        # if (c2[0]<500 or c2[0]>800):
                                        #     continue
                                        xmin=int(boxes[0])
                                        ymin=int(boxes[1])
                                        xmax=int(boxes[2])
                                        ymax=int(boxes[3])

                                        # image_np = image_np[cropped_roi.y_offset:cropped_roi.y_offset + cropped_roi.height, cropped_roi.x_offset:cropped_roi.x_offset + cropped_roi.width]
                                        image_np_crop = image_np[ymin:ymax, xmin:xmax]
                                        # output_path_crop1 = (os.path.join(self.output_path_crop, filename))
                                        ts = time.time()
                                        st = datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M_%S_%f')
                                        new_file_name=str(st) + ".jpg"
                                        output_path_crop1 = (os.path.join(self.output_path_crop, new_file_name))
                                        cv2.imwrite(output_path_crop1, image_np_crop)
                                        cv2.rectangle(image_np, c1, c2, color=(255, 0, 0), thickness=2)
                                        # cv2.putText(image_np, box_no, ((y[read_index][0]+y[read_index][2])/2, y[read_index][1]), cv2.CV_FONT_HERSHEY_SIMPLEX, 2, 255)
                                        cv2.putText(image_np, str(box_no), (int((xmin + xmax) / 2), ymin - 3), cv2.FONT_HERSHEY_SIMPLEX, .2, (0, 255, 255), lineType=cv2.LINE_AA)

                                        #########################33
                                        # saving into csv

                                        obj_id = "tl1001"
                                        obj_class = "traffic_light"

                                        width = height = 0
                                        namesplit=filename.split("_")
                                        x_loc=namesplit[0]
                                        y_loc=namesplit[-1].split(".jpg")[0]
                                        data_label = [filename, 0, width, height, obj_class, xmin, ymin, xmax, ymax, obj_id, x_loc, y_loc, box_no]
                                        self.bblabel.append(data_label)


                                    cv2.imshow('AI rectifier', image_np)
                                    ch = cv2.waitKey(1)
                                    output_path = (os.path.join(output_folder, filename))
                                    cv2.imwrite(output_path, image_np)
                    except IOError:
                        print("Existing Object Detection...")
                    except:
                        print('ERROR...object detection failed for filename: {fn}, Check file type... '.format(fn=filename), '\n')
                    else:
                        1
                            # print("Object Detected  successfully !", '\n')

                                #################################################
                                # if self.loop >= 140:
            columns = ['img_name', 'time_stamp', 'width', 'height', 'obj_class', 'xmin', 'ymin',
                       'xmax', 'ymax', 'obj_id', 'x_pose', 'y_pose',"boxno"]

            df = pd.DataFrame(self.bblabel, columns=columns)
            print("into csv file")
            # df.to_csv('template-traffic_light.csv', index=False)
            # csv_name = file_dir.split("/")[-1]
            # csv_name = file_dir + ".csv"
            csv_name = "create_csv.csv"
            df.to_csv(csv_name, index=False)



def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./cfg/yolov3.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='./weights/yolov3.pt',
                        help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    # print(opt)

    # with torch.no_grad():
    Detect(opt)




if __name__ == '__main__':
    main()