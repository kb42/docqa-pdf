from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import re

# Initialize model and processor
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def document_vqa(image_path: str, question: str) -> str:
    """
    Answers questions about document images
    Args:
        image_path: Path to document image (PDF, PNG, JPG)
        question: Natural language question about the document
    Returns:
        Answer string
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    
    # Prepare prompt with question
    task_prompt = "<s_docvqa><s_question>{}</s_question><s_answer>"
    prompt = task_prompt.format(question)
    decoder_input_ids = processor.tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    
    # Run inference
    outputs = model.generate(
        processor(image, return_tensors="pt").pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    
    # Process output
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    answer = re.search(r"<s_answer>(.*)</s_answer>", sequence).group(1)
    
    return answer

# Example usage
answer = document_vqa(
    image_path="invoice.png",
    question="What is the invoice number?"
)
print(f"Answer: {answer}")