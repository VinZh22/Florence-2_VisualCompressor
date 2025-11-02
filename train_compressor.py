import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os

from soap import SOAP

def train(model, processor, dataloader, epochs, learning_rate = 5e-5):
# Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = SOAP(model.parameters(), lr=learning_rate)
    # freeze all parameter except compression module
    for name, param in model.named_parameters():
        if 'compression_module' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
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
    model.save_pretrained("./florence_compressor_finetuned")
    processor.save_pretrained("./florence_compressor_finetuned")
    print("Model saved successfully!")
    return model, processor