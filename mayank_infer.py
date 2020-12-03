import natsort
import argparse
from models import *
from utils import *
import shutil

class Detect():
    def __init__(self, args_):
        self.opt = args_
        print(self.opt)
        img_size = (
            320,
            192) if ONNX_EXPORT else self.opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
        out, source, weights, self.half, view_img, save_txt = self.opt.output, self.opt.source, self.opt.weights, self.opt.half, self.opt.view_img, self.opt.save_txt

        self.device = select_device(device='cpu' if ONNX_EXPORT else self.opt.device)
        self.model = Darknet(self.opt.cfg, img_size)
        load_darknet_weights(self.model, weights)


        self.model.to(self.device).eval()
        self.arry_size = 6

        print("ready to process rectifier----------------------------------------------------------")
        # input_folder="/home/mayank_s/Desktop/template/farm_2/farm_2_images2_scaled"
        # input_folder = "/home/mayank_s/Downloads/cfg (2)/cfg"
        input_folder = "/home/mayank_s/datasets/nuscene/v1.0-mini/sweeps/CAM_FRONT"
        # input_folder = "/home/mayank_s/playing_git/new_2/perception_team/snowball_holiday_fix/snowball/modules/perception/src/rectifier/rectifier/weights/new"
        # input_folder="/home/mayank_s/Desktop/template/farm_1/saved_temp_img"
        # input_folder="/home/mayank_s/datasets/farminton/demo_route_complete_images/parking_lot_4"
        output_folder = "/home/mayank_s/Desktop/yolo_output"
        self.bblabel = []
        self.img_callback(input_folder, output_folder)

    # Camera image callback

    def img_callback(self, input_folder, output_folder):
            box_no = 0
            # t = time.time()
            # bridge = cv_bridge.CvBridge()
            # cv_img = bridge.imgmsg_to_cv2(image_msg, 'passthrough')
            # print("image shape is ", cv_img.shape)
            # image_np = cv_img
            if not os.path.exists(input_folder):
                print("Input folder not found")
                return 1

            # if not os.path.exists(output_folder):
            #     print("Output folder not present. Creating New folder...")
            #     os.makedirs(output_folder)
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)  # delete output folder
            os.makedirs(output_folder)

            for root, _, filenames in os.walk(input_folder):
                if (len(filenames) == 0):
                    print("Input folder is empty")
                    return 1
                time_start = time.time()
                filenames = natsort.natsorted(filenames, reverse=False)

                for filename in filenames:
                    # try:
                    print("Creating object detection for file : {fn}".format(fn=filename), '\n')

                    file_path = (os.path.join(root, filename))
                    image_np = cv2.imread(file_path, 1)

                    img, im0s = load_image(image_np, img_size=416)
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
                    print(pred)
                    # # Apply Classifier
                    # if self.classify:
                    #     pred = apply_classifier(pred, modelc, img, im0s)
                    tmp = [-1.0, 10.0, 10.0, 10.0, 10.0, 10.0]
                    if pred[0] is not None:
                        # print('prediction before', pred[0])
                        # to select only traffic light
                        # keep = torch.where(pred[0][:, 5] == 9)[0]
                        # pred[0] = pred[0][keep]
                        # print('prediction after', pred[0])
                        # Process detections
                        for loop, det in enumerate(pred):  # detections per image
                            if det is not None and len(det):
                                ############################33333
                                print(det[:, :4])
                                x=det
                                # x = x[((x[:, 2:4] > 0) & (x[:, 2:4] < 300)).all(1)]
                                x1 = x[((x[:, 0:4] > 0) & (x[:, 0:4] < 300)).all(1)]
                                ##################################3
                                tmp = -np.ones(self.arry_size * int(len(det)) + 1)
                                # Rescale boxes from img_size to im0 size
                                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                                for i, boxes in enumerate(det):
                                    box_no = box_no + 1
                                    c1, c2 = (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3]))
                                    # print(c2[0],c2[1])
                                    # if (c2[0]<500 or c2[0]>800):
                                    #     continue
                                    xmin = int(boxes[0])
                                    ymin = int(boxes[1])
                                    xmax = int(boxes[2])
                                    ymax = int(boxes[3])
                                    cv2.rectangle(image_np, c1, c2, color=(0, 255, 255), thickness=2)
                                    # cv2.putText(image_np, box_no, ((y[read_index][0]+y[read_index][2])/2, y[read_index][1]), cv2.CV_FONT_HERSHEY_SIMPLEX, 2, 255)
                                    cv2.putText(image_np, str(float(format(boxes[4], '.3f'))), (int((xmin + xmax) / 2), ymin - 3),
                                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), lineType=cv2.LINE_AA)

                                cv2.imshow('AI_rectifier', image_np)
                                ch = cv2.waitKey(1)
                    output_path = (os.path.join(output_folder, filename))
                    cv2.imwrite(output_path, image_np)

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov4_modified_tl.cfg', help='*.cfg path')
    # parser.add_argument('--cfg', type=str, default='//home/mayank_s/codebase/others/yolo/yolov4/yolov3/cfg/yolov3-1cls.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='/mayank_script/data_format_yolo/bdd100k_single_tl.names',
                        help='*.names path')
    # parser.add_argument('--weights', type=str, default='/home/mayank_s/all_model_weight/universal_weight/yolov4.conv.137', help='weights path')
    # parser.add_argument('--weights', type=str, default='/home/mayank_s/all_model_weight/aws_sync/backup/yolov4_tl_6000.weights', help='weights path')
    # parser.add_argument('--weights', type=str, default='/home/mayank_s/all_model_weight/mysnowball_weights/yolov4_tl.weights', help='weights path')
    parser.add_argument('--weights', type=str, default='/home/mayank_s/playing_git/new_2/perception_team/snowball_holiday_fix/snowball/modules/perception/src/rectifier/rectifier/weights/yolov3.wieght', help='weights path')
    parser.add_argument('--source', type=str, default='//home/mayank_s/playing_git/new_2/perception_team/snowball_holiday_fix/snowball/modules/perception/src/rectifier/rectifier/weights/new',
                        help='source')  # input file/folder, 0 for webcam
    # parser.add_argument('--source', type=str, default='/home/mayank_s/datasets/bdd/training_set/one_class/bdd100k_images/bdd100k/images/100k/train', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='/home/mayank_s/Desktop/yolo_output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment',default=True, action='store_true', help='augmented inference')
    opt = parser.parse_args()
    # print(opt)

    # with torch.no_grad():
    Detect(opt)


if __name__ == '__main__':
    main()
