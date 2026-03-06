from tqdm import tqdm
from PIL import Image
import os
import io
import json
from typing_extensions import List,Optional
import numpy as np
from collections import defaultdict

def dump_jsonlines(data, path):
    with open(path, 'w') as f:
        f.write('\n'.join([json.dumps(item, ensure_ascii=False) for item in data]))
def load_jsonlines(jsonl_path):
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading jsonl file"):
            try:
                d = json.loads(line)
                data.append(d)
            except:
                continue
    return data


def calculate_iou(box1, box2):

    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area1 = max(0.0, (x2_1 - x1_1)) * max(0.0, (y2_1 - y1_1))
    area2 = max(0.0, (x2_2 - x1_2)) * max(0.0, (y2_2 - y1_2))
    union = area1 + area2 - inter_area
    if union == 0:
        return 0.0
    return inter_area / union

def check_acc(pred_bbox,gt_bbox,task_type):
    if task_type != "reject":
        iou = calculate_iou(pred_bbox,gt_bbox)
    else:
        if pred_bbox == [0,0,0,0]:
            iou = 1.0 #RejAcc = 1
        else:
            iou = 0.0 #RejAcc = 0
    return iou



def calculate_macc_from_iou(all_result_data):
    category_metrics = defaultdict(lambda: {
        'scores': [], 
        'reject_correct': 0, 
        'reject_total': 0
    })

    iou_thresholds = np.arange(0.5, 0.95, 0.05)

    for data in all_result_data:
        try:
            task_type = data.get("task_type")
            iou = data.get("iou")

            if task_type is None or iou is None:
                continue
            
            if task_type == 'reject':
                category_metrics[task_type]['reject_total'] += 1
                if iou == 1.0:
                    category_metrics[task_type]['reject_correct'] += 1
            else:
                correct_count = np.sum(iou >= iou_thresholds)
                
                accuracy_score = correct_count / len(iou_thresholds)
                
                category_metrics[task_type]['scores'].append(accuracy_score)

        except json.JSONDecodeError:
            continue

    final_results = {}
    for task_type, metrics in category_metrics.items():
        if task_type == 'reject':
            if metrics['reject_total'] > 0:
                accuracy = metrics['reject_correct'] / metrics['reject_total']
                final_results[task_type] = accuracy
            else:
                final_results[task_type] = 0.0 
        else:
            if metrics['scores']:
                mean_acc = np.mean(metrics['scores'])
                final_results[task_type] = mean_acc
            else:
                final_results[task_type] = 0.0
    return final_results

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Inference script for Qwen-2.5-VL")
    parser.add_argument(
        "--prediction_file",
        type=str,
        default = "pred_results/results_qwen3.jsonl",
        help="Path to the prediction file",
    )


    return parser.parse_args()

if __name__ == "__main__":

    args = get_args()    
    data_path = args.prediction_file
    all_data = load_jsonlines(data_path)

    results = []

    for data in tqdm(all_data):
        new_result = data.copy()
        iou = check_acc(data["pred_bbox"], data["bbox"],data["task_type"])
        new_result["iou"] = iou
        results.append(new_result)
    
    acc_result = calculate_macc_from_iou(results)
    if acc_result:
        for category, score in sorted(acc_result.items()): 
            print(f"  - {category:<10}: {score*100:.2f}")
