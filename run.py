import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os

from soap import SOAP
import argparse

class FlorenceDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, prompt_dir, processor):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.prompt_dir = prompt_dir
        self.processor = processor
        
        # Load all files in __init__
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Load all prompts and annotations in __init__
        self.prompts = {}
        self.annotations = {}
        with open(os.path.join(self.prompt_dir, 'questions.json'), 'r', encoding='utf-8') as pf:
            self.prompts = json.load(pf)
        
        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
        # Get image file directly by index
        image_file = self.image_files[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        
        # Get prompt and annotation from pre-loaded dictionaries
        prompt = self.prompts[image_file]
        annotation = self.annotations[image_file]
        
        # Process inputs
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        labels = self.processor.tokenizer(text=annotation, return_tensors="pt", padding=True, return_token_type_ids=False)
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'labels': labels['input_ids'].squeeze(0)
        }

def main(args):
    # Configuration
    model_name = "./" + args.model_name
    image_dir = args.image_dir
    annotation_dir = args.annotation_dir
    prompt_dir = args.prompt_dir
    epochs = 3
    batch_size = 4
    learning_rate = 5e-5
    

    # Load model and processor
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, trust_remote_code=True)
    
    # Create dataset and dataloader
    dataset = FlorenceDataset(image_dir, annotation_dir, prompt_dir, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = SOAP(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    
    # Save the fine-tuned model
    model.save_pretrained("./florence_finetuned")
    processor.save_pretrained("./florence_finetuned")
    print("Model saved successfully!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-tune Florence-2 model')
    parser.add_argument('--model_name', type=str, default='Florence-2-base', help='Model name')
    parser.add_argument('--compression_mode', type=str, default=None, help='Compression mode: avg_pool or learnable_pool')
    parser.add_argument('--compression_factor', type=int, default=None, help='Compression factor (pool size)')
    parser.add_argument('--compression_stage', type=int, default=None, help='Layer index to apply compression at')
    parser.add_argument('--compression_sorted', action='store_true', help='Whether to sort inputs for learnable pooling')

    # Training if compression if learnable pooling
    parser.add_argument('--prompt_dir', type=str, default='./data/prompts', help='Directory containing prompts')
    parser.add_argument('--annotation_dir', type=str, default='./data/annotations', help='Directory containing annotations')
    parser.add_argument('--image_dir', type=str, default='./data/images', help='Directory containing images')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./florence_finetuned', help='Output directory for saved model')

    # Inference arguments (testing purposes)
    parser.add_argument('--test_image_dir', type=str, default='./data/test_images', help='Directory containing test images')
    parser.add_argument('--test_prompt', type=str, default='<CAPTION>', help='Prompt for inference')

    ## We are going to use VQA as basis for training and testing, so the dataset format are going to match their's.

    
    args = parser.parse_args()
    main(args)