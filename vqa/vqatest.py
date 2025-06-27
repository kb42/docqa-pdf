from datasets import load_dataset
import json
from tqdm import tqdm
from src.vqa import document_vqa

# Load dataset (replace path with your download)
dataset = load_dataset("imagefolder", data_dir="DocVQA_Val", split="train")

# Load annotations
with open("DocVQA_Val/val_v1.0.json") as f:
    annotations = json.load(f)["data"]

# Run evaluation
correct = 0
total = 0
results = []

for i in tqdm(range(min(10, len(dataset)))):  # Test first 10 samples
    item = dataset[i]
    annotation = annotations[i]
    
    # Run VQA
    answer = document_vqa(
        image_path=item["image"],
        question=annotation["question"]
    )
    
    # Check accuracy
    correct_answers = [ans.lower() for ans in annotation["answers"]]
    is_correct = answer.lower() in correct_answers
    correct += int(is_correct)
    total += 1
    
    results.append({
        "image": annotation["image"],
        "question": annotation["question"],
        "predicted": answer,
        "expected": correct_answers,
        "correct": is_correct
    })

# Print summary
print(f"\nAccuracy: {correct/total:.2%}")
for res in results:
    print(f"\nQ: {res['question']}")
    print(f"A: {res['predicted']} (Expected: {', '.join(res['expected'])})")
    print(f"Correct: {'✓' if res['correct'] else '✗'}")