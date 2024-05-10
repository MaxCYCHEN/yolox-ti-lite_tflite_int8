import argparse
import os
import time
import json
import io
import contextlib
import itertools
from tqdm import tqdm

from tabulate import tabulate

import cv2
import numpy as np
import tensorflow.lite as tflite

from utils import COCO_CLASSES, multiclass_nms_class_aware, preprocess, postprocess, vis, yolofastest_preprocess, yolofastest_postprocess
from pycocotools.coco import COCO

# ToDo, tmp global var
per_class_mAP = True

def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('-m', '--model', required=True, help='path to .tflite model')
        parser.add_argument('-i', '--img', help='path to image file')
        parser.add_argument('-v', '--val', default='..\edgeai-yolox\datasets\coco', help='path to validation dataset')
        #parser.add_argument('-v', '--val', default='datasets\coco_test_sp', help='path to validation dataset')
        parser.add_argument('-o', '--out-dir', default='tmp/tflite', help='path to output directory')
        parser.add_argument('-s', '--score-thr', type=float, default=0.3, help='threshould to filter by scores')
        return parser.parse_args()

class coco_format_dataset():
    def __init__(
        self,
        data_dir="datasets\coco_test_sp",
        val_json_file="instances_val2017.json",
        img_size=(416, 416),
    ):
        self.img_size = img_size # This val isn't used in validation
        self.data_dir = data_dir
        self.img_dir_name = "val2017"
        self.annotation_dir_name = "annotations"
        self.gd_annotation_file = os.path.join(data_dir, self.annotation_dir_name, val_json_file)
        self.coco = COCO(self.gd_annotation_file)
        self.img_ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        self.annotations = self._load_coco_annotations()

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.img_ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (img_info, file_name)    

    def _load_image(self, index):
        file_name = self.annotations[index][1]
        img_file = os.path.join(self.data_dir, self.img_dir_name, file_name)
        img = cv2.imread(img_file)
        assert img is not None

        return img

    def pull_item(self, index):
        id_ = self.img_ids[index]

        img_info, file_name = self.annotations[index]
        img = self._load_image(index)

        return img, file_name, img_info, np.array([id_])    

    def __getitem__(self, index):
        img, file_name, img_info, img_id = self.pull_item(index)
        #if self.preproc is not None:
        #    img, target = self.preproc(img, target, self.input_dim)
        return img, file_name, img_info, img_id    
    
    def per_class_mAP_table(self, coco_eval, class_names=COCO_CLASSES, headers=["class", "AP"], colums=6):
        per_class_mAP = {}
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]
    
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            per_class_mAP[name] = float(ap * 100)
    
        num_cols = min(colums, len(per_class_mAP) * len(headers))
        result_pair = [x for pair in per_class_mAP.items() for x in pair]
        row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
        table_headers = headers * (num_cols // len(headers))
        table = tabulate(
            row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
        )
        return table
    
    def evaluate_prediction(self, data_dict):
           
            print("Evaluate in main process...")
    
            annType = ["segm", "bbox", "keypoints"]
    
            info = "\n"
    
            # Evaluate the Dt (detection) json comparing with the ground truth
            if len(data_dict) > 0:
                cocoGt = self.coco
    
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            
                try:
                    from yolox.layers import COCOeval_opt as COCOeval
                except ImportError:
                    from pycocotools.cocoeval import COCOeval
    
                    print("Use standard COCOeval.")
    
                cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
                cocoEval.evaluate()
                cocoEval.accumulate()
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                info += redirect_string.getvalue()
                if per_class_mAP:
                    info += "per class mAP:\n" + self.per_class_mAP_table(cocoEval)
                return cocoEval.stats[0], cocoEval.stats[1], info
            else:
                return 0, 0, info
    
    def xyxy2xywh(self, bboxes):
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
        return bboxes
    
    def convert_to_coco_format(self, outputs, model_img_size, info_imgs, ids):
            data_list = []
            for (output, ori_img, img_id) in zip(
                outputs, info_imgs, ids
            ):
                if output is None:
                    continue
                
                bboxes = output[:, 0:4]
    
                # preprocessing: resize to original
                #img_h = ori_img[0]
                #img_w = ori_img[1]
                #scale = min(
                #    model_img_size[0] / float(img_h), model_img_size[1] / float(img_w)
                #)
                #bboxes = bboxes / scale
                bboxes = self.xyxy2xywh(bboxes)
    
                cls = output[:, 5]
                scores = output[:, 4]
                for ind in range(bboxes.shape[0]):
                    #label = COCO_CLASSES[int(cls[ind])] # Update your class
                    label = self.class_ids[int(cls[ind])]
                    pred_data = {
                        "image_id": int(img_id),
                        "category_id": label,
                        "bbox": bboxes[ind].tolist(),
                        "score": scores[ind].item(),
                        "segmentation": [],
                    }  # COCO json format
                    data_list.append(pred_data)

                
            return data_list


def main():
    # reference:
    # https://github.com/PINTO0309/PINTO_model_zoo/blob/main/132_YOLOX/demo/tflite/yolox_tflite_demo.py

    args = parse_args()

    # setup dataset
    data_list = []
    my_dataset = coco_format_dataset(data_dir=args.val)

    # prepare model
    interpreter = tflite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # model info
    input_dtype = input_details[0]['dtype']
    input_scale = input_details[0]['quantization'][0]
    input_zero = input_details[0]['quantization'][1]
    print("Model Shape: {} {} Model Dtype: {}"
    .format(input_details[0]['shape'][1], input_details[0]['shape'][2], input_dtype))
    print("Model input Scale: {} Model input Zero point: {}"
    .format(input_scale, input_zero))

    output_dtype = output_details[0]['dtype']
    output_scale = output_details[0]['quantization'][0]
    output_zero  = output_details[0]['quantization'][1]
    print("Model output Shape: {} {} Model output Dtype: {}"
    .format(output_details[0]['shape'][0], output_details[0]['shape'][1], output_dtype))
    print("Model output Scale: {} Model output Zero point: {}"
    .format(output_details[0]['quantization'][0], output_details[0]['quantization'][1]))

    input_shape = input_details[0]['shape']
    b, h, w, c = input_shape
    model_img_size = (h, w)
    

    for cur_iter, (origin_img, file_name, info_imgs, ids) in enumerate(tqdm(my_dataset)):
        origin_img_size = (origin_img.shape[0], origin_img.shape[1])

        #img, ratio = preprocess(origin_img, model_img_size)
        img = yolofastest_preprocess(origin_img, model_img_size)
        
        img = img[np.newaxis].astype(np.float32)  # add batch dim
        
        if input_dtype == np.int8:
            img = img - 128
            img = img.astype(np.int8)
        
        #    #print("input int8 converting:")
        #    img = img / input_scale + input_zero
        #    img = img.astype(np.int8)
        else:
            img = (img)/255

        
        # run inference
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
    
        outputs_1 = interpreter.get_tensor(output_details[0]['index'])[0]  # remove batch dim
        outputs_2 = interpreter.get_tensor(output_details[1]['index'])[0]  # remove batch dim
        # int8
        if output_dtype == np.int8:
            outputs_1 = output_scale * (outputs_1.astype(np.float32) - output_zero)
            outputs_2 = output_details[1]['quantization'][0] * (outputs_2.astype(np.float32) - output_details[1]['quantization'][1])
        #print(outputs_1.shape)
        #print(outputs_2.shape)
        #print(len(COCO_CLASSES))
        ori_img_h = origin_img.shape[0]
        ori_img_w = origin_img.shape[1]
        anchor1 = [12, 18,  37, 49,  52,132]
        anchor2 = [115, 73, 119,199, 242,238]
        num_boxs = len(anchor1)/2
        class_num = int((outputs_1.shape[2] / num_boxs) - 5)
        assert class_num==len(COCO_CLASSES), "The classes doesn't match with yolofastest output"

        
        detection_res_list_0 = yolofastest_postprocess(outputs_1, anchor1, class_num, model_img_size, origin_img_size)
        detection_res_list_1 = yolofastest_postprocess(outputs_2, anchor2, class_num, model_img_size, origin_img_size)
        detection_res_list_0.extend(detection_res_list_1)


        boxes_xyxy = np.ones((len(detection_res_list_0), 4))
        scores = np.zeros((len(detection_res_list_0), class_num))
        idx = 0
        for det in detection_res_list_0:
            boxes_xyxy[idx, 0] = det['x'] - det['w'] / 2.0
            boxes_xyxy[idx, 1] = det['y'] - det['h'] / 2.0
            boxes_xyxy[idx, 2] = det['x'] + det['w'] / 2.0
            boxes_xyxy[idx, 3] = det['y'] + det['h'] / 2.0
            scores[idx, :] = det['sig']
            idx+=1

        dets = multiclass_nms_class_aware(boxes_xyxy, scores, nms_thr=0.65, score_thr=0.01)
        
        #print("Inference: {0:.2f}ms".format(inference_time))
        #dets = np.concatenate(([[1.02412483e+02, 8.34888458e+01, 2.73776947e+02, 3.89458191e+02, 8.36849928e-01, 0.00000000e+00]], dets))
        #print(dets)
        #print(model_img_size, info_imgs, ids)

        # single coco mAP eval
        data_list.extend(my_dataset.convert_to_coco_format([dets], model_img_size, [info_imgs], ids))
        
        # visualize and save
        #if dets is None:
        #    print("no object detected.")
        #else:
        #    det_ori_box = dets[:, :4]
        #    final_boxes, final_scores, final_cls_inds = det_ori_box, dets[:, 4], dets[:, 5]
        #    origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
        #                     conf=args.score_thr, class_names=COCO_CLASSES)    
        #os.makedirs(args.out_dir, exist_ok=True)
        #cv2.imwrite(r'C:\Users\USER\Desktop\ML\yolox-ti-lite_tflite\tmp\tflite\outputxxx.jpg', origin_img)

    # coco mAP eval
    *_, summary = my_dataset.evaluate_prediction(data_list)
    print(summary)


    


if __name__ == '__main__':
    main()
