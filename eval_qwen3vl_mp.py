import torch
from tqdm import tqdm
from PIL import Image
import multiprocessing
import os
import torch
import math
import io
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import json
from typing_extensions import List,Optional
import re
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import numpy as np
from collections import defaultdict


# Preparation for inference

MIN_PIXEL=4*32*32
MAX_PIXEL = 1024*32*32


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
import base64


def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int =MIN_PIXEL, max_pixels: int =MAX_PIXEL):
    
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar
def image_to_base64(img, format="PNG"):
    sh, sw = smart_resize(img.size[1], img.size[0],factor = 32)
    img = img.resize((sw, sh))
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    base64_qwen = f"data:image;base64,{base64_str}"
    return base64_qwen, sh, sw

def resize_bbox(bbox,original_width, original_height):

    x1_pred, y1_pred, x2_pred, y2_pred = bbox
    x1_orig = x1_pred * (original_width / 1000)
    y1_orig = y1_pred * (original_height / 1000)
    x2_orig = x2_pred * (original_width / 1000)
    y2_orig = y2_pred * (original_height / 1000)

    final_x1 = max(0, min(x1_orig, original_width - 1))
    final_y1 = max(0, min(y1_orig, original_height - 1))
    final_x2 = max(0, min(x2_orig, original_width - 1))
    final_y2 = max(0, min(y2_orig, original_height - 1))
    return [final_x1,final_y1,final_x2,final_y2]

def check_acc(pred_bbox,gt_bbox,task_type):
    if task_type != "reject":
        iou = calculate_iou(pred_bbox,gt_bbox)
    else:
        if pred_bbox == [0,0,0,0]:
            iou = 1.0 #RejAcc = 1
        else:
            iou = 0.0 #RejAcc = 0
    return iou


def extract_bbox_from_response(response_text: str) -> Optional[List[float]]:
    if not isinstance(response_text, str) or not response_text:
        return None

    def _parse_and_validate_bbox(content_str: str) -> Optional[List[float]]:

        try:
            bbox = json.loads(f"[{content_str}]")
            if (isinstance(bbox, list) and
                len(bbox) == 4 and
                all(isinstance(n, (int, float)) for n in bbox)):
                return [float(n) for n in bbox]
        except json.JSONDecodeError:
            pass
        
        try:
            numbers_str = content_str.split(',')
            if len(numbers_str) == 4:
                bbox = [float(n.strip()) for n in numbers_str]
                return bbox
        except (ValueError, IndexError):
            pass
            
        return None

    all_contents = re.findall(r'\[(.*?)\]', response_text)

    if not all_contents:
        return None

    for content_inside_brackets in reversed(all_contents):
        bbox = _parse_and_validate_bbox(content_inside_brackets)
        if bbox is not None:
            return bbox

    return None

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


def get_pred_bbox(response,width,height):
    ori_pred_bbox = extract_bbox_from_response(response)
    if ori_pred_bbox is None:
        pred_bbox = [0,0,0,0]
    else:
        pred_bbox = resize_bbox(ori_pred_bbox,width,height)
    return pred_bbox

def process_data(data):
    global worker_model, worker_device,worker_processor
    current_file_path = os.path.abspath(__file__)
    image_path_dir = os.path.join(os.path.dirname(current_file_path),"images")
    try:
        
        image_path = os.path.join(image_path_dir,data["image_path"])
        expression = data["expression"]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image":  Image.open(image_path).convert("RGB"),
                        "min_pixels": MIN_PIXEL, "max_pixels": MAX_PIXEL,
                    },
                    {"type": "text", "text": f"Locate {expression}, output its bbox coordinates using JSON format."},
                ],
            }
        ]

        inputs = worker_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = {
            key: value.to(worker_device) 
            for key, value in inputs.items()
        }
        generated_ids = worker_model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        response = worker_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        pred_bbox = get_pred_bbox(response[0],data["width"],data["height"]) #actual pred bbox or [0,0,0,0] for error
        iou = check_acc(pred_bbox, data["bbox"],data["task_type"])
        new_result = data
        new_result["iou"] = iou
        new_result["pred_bbox"] = pred_bbox
        new_result["response"] = response[0]
        print(new_result)
        return new_result
        
    except Exception as e:
        print(f"error in  {os.getpid()} : {e}")
        return None

def init_worker(model_path):
    global worker_model, worker_device,worker_processor
    
    worker_id = multiprocessing.current_process()._identity[0] - 1
    
    visible_devices_str = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    visible_devices = [int(x) for x in visible_devices_str.split(',')]
    device_idx = worker_id % len(visible_devices)
    
    worker_device = torch.device(f"cuda:{device_idx}")
        
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype="auto")
    worker_model = model.to(worker_device)
    worker_processor = AutoProcessor.from_pretrained(model_path)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True) 
    
    data_path = "annotations_6000.jsonl"
    all_data = load_jsonlines(data_path)
    model_path = "Qwen/Qwen3-VL-8B-Instruct"


    NUM_PROCESSES = 16

    init_args = (model_path,)
    results = []
    with multiprocessing.Pool(processes=NUM_PROCESSES, initializer=init_worker, initargs=init_args) as pool:
        for result_item in tqdm(pool.imap_unordered(process_data, all_data), total=len(all_data)):
            if result_item is not None:
                results.append(result_item)
    
    
    acc_result = calculate_macc_from_iou(results)
    if acc_result:
        for category, score in sorted(acc_result.items()): 
            print(f"  - {category:<10}: {score*100:.2f}")
    dump_jsonlines(results, "results_qwen3.jsonl")
    