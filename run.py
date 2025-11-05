import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import json
import os

from .train_compressor import train
from .data import VQADataset
from .inference import run_example, evaluate

from soap import SOAP
from datetime import datetime
import argparse


def main(args):
    # Configuration
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    model_name = "./" + args.model_name
    epochs = 3
    batch_size = 4
    learning_rate = 5e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading model and processor...")
    # Load model and processor
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.compression_mode = args.compression_mode
    config.compression_factor = args.compression_factor
    config.compression_stage = args.compression_stage
    config.compression_sorted = args.compression_sorted

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True).to(device)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, trust_remote_code=True).to(device)
    print("Loading dataset...")
    # Create dataset and dataloader
    dataset = VQADataset(
        pct_data = args.pct_data,
        data_path = args.data_dir,
        annotation_folder = args.annotation_folder,
        question_folder = args.prompt_folder,
        image_folder = args.image_folder,
        image_prefix = args.image_prefix,
    )

    def collate_fn(batch):
        questions, answers, images, _ = zip(*batch)
        inputs = processor(
            text=list(questions), images=list(images), return_tensors="pt", padding=True
        ).to(device)
        return inputs, answers

    train_size = int(0.8 * len(dataset))  # 80% train
    test_size = len(dataset) - train_size  # 20% test
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn,batch_size=batch_size, shuffle=False)
    print(f"Dataset split into {train_size} training and {test_size} testing samples.")
    # TODO
    if args.compression_mode =='learnable_pool':
        print("Starting training for a learnable pool...")
        # Train the model with compression
        model, processor = train(
            model=model,
            processor=processor,
            dataloader=train_dataloader,
            epochs=epochs,
            learning_rate=learning_rate
        )
        # Save the fine-tuned model
        save_path = "./Results_learn"
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(os.path.join(save_path, timestamp+"_model"))
        processor.save_pretrained(os.path.join(save_path, timestamp+"_processor"))
        print("Model saved successfully!")
    
    print("Running inference on test images...")
    # Inference example on test images
    answers, avg_levenshtein_similarity = evaluate(model, processor, test_dataloader)

    print("Evaluation Results:")
    print(f"Average Levenshtein Similarity: {avg_levenshtein_similarity:.4f}")
    print("Sample Answers:")
    for i, answer in enumerate(answers[:5]):
        print(f"Answer {i+1}: {answer}")
    
    # Save the answers to a file
    print("Saving answers to file...")
    output_file = os.path.join(args.output_dir, timestamp + "_test_answers.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(answers, f)

    print("Done.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-tune Florence-2 model')
    parser.add_argument('--model_name', type=str, default='Florence-2-base', help='Model name')
    parser.add_argument('--compression_mode', type=str, choices=['avg_pool', 'learnable_pool', 'none'], default=None, help='Compression mode: avg_pool or learnable_pool')
    parser.add_argument('--compression_factor', type=int, default=None, help='Compression factor (pool size)')
    parser.add_argument('--compression_stage', type=int, default=None, help='Layer index to apply compression at')
    parser.add_argument('--compression_sorted', action='store_true', help='Whether to sort inputs for learnable pooling')

    # File paths and data settings
    parser.add_argument('--data_dir', type=str, default='./Data', help='Directory containing dataset')
    parser.add_argument('--prompt_folder', type=str, default='v2_Questions', help='Folder containing prompts')
    parser.add_argument('--annotation_folder', type=str, default='v2_Annotations', help='Folder containing annotations')
    parser.add_argument('--image_folder', type=str, default='train2014_image', help='Folder containing images')
    parser.add_argument('--pct_data', type=int, default=1, help='Percentage of data to use for training (e.g., 1, 5, 10, 100)')
    parser.add_argument('--image_prefix', type=str, default='COCO_train2014_', help='Prefix for image filenames')

    # Training if compression if learnable pooling
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')

    parser.add_argument('--output_dir', type=str, default='./florence_predictions', help='Output directory for saved model')

    # Inference arguments (testing purposes)
    parser.add_argument('--test_image_dir', type=str, default='./data/test_images', help='Directory containing test images')
    parser.add_argument('--test_prompt', type=str, default='<CAPTION>', help='Prompt for inference')

    ## We are going to use VQA as basis for training and testing, so the dataset format are going to match their's.

    
    args = parser.parse_args()
    main(args)