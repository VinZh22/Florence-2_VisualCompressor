"""
Inspired from data.py of https://github.com/andimarafioti/florence2-finetuning
Adapted to VQA data (not data from HugginFace datasets)
"""


import json
import os
import pandas as pd
from datasets import get_dataset_config_names, load_dataset, load_from_disk
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import random

class BaseDataset(Dataset):
    def __init__(self, split):
        self.name = "BaseDataset"
        self.data = []
        self.task_prompt = ""

    def __len__(self):
        return len(self.data)

    def correct_casing_finqa(self, text, is_question=False):
        if text and text[0].islower():
            text = text.capitalize()
        if not text.endswith(".") and not is_question:
            text += "."
        if not text.endswith("?") and is_question:
            text += "?"
        return text

class VQADataset(BaseDataset):
    def __init__(self, pct_data, data_path, 
                 annotation_folder = "v2_Annotations_Train", 
                 question_folder = "v2_Questions_Train", 
                 image_folder = "train2014_image", image_prefix = "COCO_train2014_", 
                 ):
        super().__init__()
        self.data_path = data_path
        self.annotation_folder = annotation_folder
        self.question_folder = question_folder
        self.image_folder = image_folder
        self.image_prefix = image_prefix
        self.pct_data = pct_data

        self.image_id_len = 12 # Constant for now, used to pad image ids for their name (change and make it parameter if it changes)
        self.name = "VQADataset"
        self.task_prompt = "<VQA>"
        self._load_data()
    
    def _load_data(self):
        annotation_filename = f"annotations_sample_{self.pct_data}pct.json" if self.pct_data < 100 else "annotations.json"
        question_filename = f"questions_sample_{self.pct_data}pct.json" if self.pct_data < 100 else "questions.json"
        
        annotation_path = os.path.join(self.data_path, self.annotation_folder, annotation_filename)
        question_path = os.path.join(self.data_path, self.question_folder, question_filename)

        with open(annotation_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        
        with open(question_path, "r", encoding="utf-8") as f:
            questions = json.load(f)["questions"]
        
        question_dict = {q["question_id"]: q for q in questions}
        
        for ann in tqdm(annotations, desc=f"Loading {self.name}"):
            question_id = ann["question_id"]
            question_data = question_dict.get(question_id, {})
            if not question_data:
                continue
            
            image_id = question_data["image_id"]
            image_id = str(image_id).zfill(self.image_id_len)
            image_path = os.path.join(self.data_path, self.image_folder, f"{self.image_prefix}{image_id}.jpg")

            question_text = self.correct_casing_finqa(question_data["question"], is_question=True)
            answer_text = self.correct_casing_finqa(ann["multiple_choice_answer"], is_question=False)
            
            self.data.append({
                "image_path": image_path,
                "question": question_text,
                "answer": answer_text,
                "question_id": question_id
            })
    
    def __getitem__(self, idx):
        question = self.data[idx]["question"]
        answer = self.data[idx]["answer"]
        image_path = self.data[idx]["image_path"]

        question_prompt = self.task_prompt + question
        image = Image.open(image_path).convert("RGB")
        return question_prompt, answer, image

    def get_question_id(self, idx):
        return self.data[idx]["question_id"]
