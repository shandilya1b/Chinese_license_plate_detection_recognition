import os
from detect_plate import load_model, detect_Recognition_plate, draw_result
from plate_recognition.plate_rec import init_model
import shutil
import argparse
from tqdm import tqdm
import cv2

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object-weights', nargs='+', type=str, default='object.pt', help='model path or triton URL')
    parser.add_argument('--char-weights', nargs='+', type=str, default='char.pt', help='model path or triton URL')
    parser.add_argument('--out-dir', default='out', help='path to output folder')
    parser.add_argument('--dataset-dir', help='path to dataset to run inference over')
    parser.add_argument('--object-imgsz', '--object-img', '--object-img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--char-imgsz', '--char-img', '--char-img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--object-conf-thres', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--object-iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--char-conf-thres', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--char-iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    # opt.object_imgsz *= 2 if len(opt.object_imgsz) == 1 else 1  # expand
    # opt.char_imgsz *= 2 if len(opt.char_imgsz) == 1 else 1  # expand
    opt.object_imgsz = opt.object_imgsz[0]
    opt.char_imgsz = opt.char_imgsz[0]
    print('object detection image size: ', opt.object_imgsz)
    print('character recognition image size: ', opt.char_imgsz)
    opt.char_weights = opt.char_weights[0]
    return opt

if __name__ == '__main__':
    opt = parse_opts()
    dataset_dir = opt.dataset_dir
    print('Loading data from: ', dataset_dir)
    dest_path=opt.out_dir
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)    
    os.mkdir(dest_path)
    
    device = opt.device
    detect_model = load_model(opt.object_weights, device)
    print(opt.char_weights)
    plate_rec_model=init_model(device,opt.char_weights,is_color=True)      #初始化识别模型
    total = sum(p.numel() for p in detect_model.parameters())
    total_1 = sum(p.numel() for p in plate_rec_model.parameters())
    print("detect params: %.2fM,rec params: %.2fM" % (total/1e6,total_1/1e6))
    
    labels = os.listdir(dataset_dir)

    for item in labels:
        if os.path.isdir(os.path.join(dataset_dir, item)):
            label_dir = os.path.join(dataset_dir, item)
            out_dir = os.path.join(dest_path, item)
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.mkdir(out_dir)
            plates_out_dir = os.path.join(out_dir, 'plates')
            if os.path.exists(plates_out_dir):
                shutil.rmtree(plates_out_dir)
            os.mkdir(plates_out_dir)

            for image_name in tqdm(os.listdir(label_dir)):

                image_path = os.path.join(label_dir, image_name)
                img = cv2.imread(image_path)
                dict_list = detect_Recognition_plate(detect_model, img, device,plate_rec_model,opt.object_imgsz,is_color=True)#检测以及识别车牌
                draw_img = draw_result(img.copy(), dict_list)
                cv2.imwrite(os.path.join(out_dir, image_name), draw_img)
