import torch
from tqdm import tqdm

from .metrics import average_normalized_levenshtein_similarity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_example(model, processor, task_prompt, query, image):
    """
    An example function that processes input and image.
    """
    model.to(device)

    prompt = task_prompt + query

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    # Run the model on the processed input and image
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3,
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

    return parsed_answer


def run_batch(model, processor, inputs):
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
    )
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
    return generated_texts

def evaluate(model, processor, test_loader):
    model.to(device)
    model.eval()
    all_predictions = []
    task_prompt = "<VQA>" # Hardcoded for now

    ##TO FINISH
    predicted_answers = []
    ground_truth = []

    for inputs, batch_answers in tqdm(test_loader, desc="Evaluating"):
        generated_texts = run_batch(inputs)

        for generated_text, answers in zip(generated_texts, batch_answers):
            parsed_answer = processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(
                    inputs["pixel_values"].shape[-2],
                    inputs["pixel_values"].shape[-1],
                ),
            )
            predicted_answers.append(parsed_answer[task_prompt].replace("<pad>", ""))
            ground_truth.append(answers)
            # print("Ans:", parsed_answer[task_prompt])
            # print("GT:", answers)

    avg_levenshtein_similarity = average_normalized_levenshtein_similarity(
        ground_truth, predicted_answers
    )
    return answers, avg_levenshtein_similarity

def show_example_results(model, processor, examples):
    model.to(device)
    model.eval()

    for example in examples:
        image = example["image"]
        query = example["query"]
        task_prompt = example["task_prompt"]

        answer = run_example(model, processor, task_prompt, query, image)
        print(f"Query: {query}")
        print(f"Answer: {answer}")
        print("-" * 50)