from docling.document_converter import DocumentConverter
from logging_config import setup_logging

logger = setup_logging(__name__)

def extract_text_from_image(image_path):
    try:
        source = image_path
        converter = DocumentConverter()
        result = converter.convert(source)
        extracted_text = result.document.export_to_markdown()
        if not extracted_text:
            logger.error("No text was extracted from the image")
            return None
        return extracted_text
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        return None

from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM


# Load model and tokenizer
model_id = "OpenVINO/Phi-3-mini-4k-instruct-int4-ov"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = OVModelForCausalLM.from_pretrained(model_id, device="CPU")

def validation_ai(text):
    if not text:
        return "Error: No text provided for validation"
    
    # Define system and user prompts
    system_prompt = """You are a validator comparing billing and shipping details.
    Please check if the provided information contains matching billing and shipping details.
    Respond with 'MATCH' if they are the same, or 'MISMATCH' if they differ."""
    
    user_prompt = f"Information: {text}"

    # Format the prompt using the chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
        # Decode the output
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return response
    except Exception as e:
        logger.error(f"Error in validation: {str(e)}")
        return f"Error during validation: {str(e)}"

if __name__ == "__main__":

    image_path = "images/customer page.jpg"
    
    # Verify file exists
    import os
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        exit(1)
        
    text = extract_text_from_image(image_path)
    if text:
        logger.info(f"Extracted text: {text}")
        res = validation_ai(text)
        logger.info(f"Validation result: {res}")
        print(res)
    else:
        print("Failed to extract text from image")
