import torch
from tqdm import tqdm

import pdb
from metrics import average_normalized_levenshtein_similarity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_standard_accuracy(predictions, group_truths):
    """
    Computes the standard accuracy for VQA tasks.
    If more than 3 annotations: +1
    If 2 annotations: +0.66
    If 1 annotation: +0.33
    0 otherwise
    """
    correct = 0

    for pred, truths in zip(predictions, group_truths):
        match_count = sum([1 for truth in truths if pred.strip().lower() == truth.strip().lower()])
        if match_count >= 3:
            correct += 1
        elif match_count == 2:
            correct += 0.66
        elif match_count == 1:
            correct += 0.33
    accuracy = correct / len(predictions)
    return accuracy


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

def evaluate(model, processor, test_loader, test_dataset):
    model.to(device)
    model.eval()
    all_predictions = []
    task_prompt = "<VQA>" # Hardcoded for now

    ##TO FINISH
    questions_asked = []
    predicted_answers = []
    ground_truth = []
    group_truths = []
    idx = 0
    for inputs, batch_answers in tqdm(test_loader, desc="Evaluating"):
        generated_texts = run_batch(model, processor, inputs)
        for inp in inputs["input_ids"]:
            questions_asked.append(processor.decode(inp, skip_special_tokens=True))
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
            group_truths.append(test_dataset.data[idx]['answers'])
            idx += 1
            # print("Ans:", parsed_answer[task_prompt])
            # print("GT:", answers)

    avg_levenshtein_similarity = average_normalized_levenshtein_similarity(
        ground_truth, predicted_answers
    )
    standard_accuracy = compute_standard_accuracy(predicted_answers, group_truths)

    return predicted_answers, ground_truth, avg_levenshtein_similarity, standard_accuracy

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