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


def calculate_iou(boxA: List[int], boxB: List[int]) -> float:

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    unionArea = float(boxAArea + boxBArea - interArea)
    
    iou = interArea / unionArea if unionArea > 0 else 0.0
    return iou

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
        new_result = data
        iou = check_acc(data["pred_bbox"], data["bbox"],data["task_type"])
        new_result["iou"] = iou
        # print(new_result)
        results.append(new_result)
    
    acc_result = calculate_macc_from_iou(results)
    if acc_result:
        for category, score in sorted(acc_result.items()): 
            print(f"  - {category:<10}: {score*100:.2f}")
