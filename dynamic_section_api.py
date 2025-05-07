from difflib import SequenceMatcher
from fastapi import FastAPI, File, UploadFile
from logging_config import setup_logging
import torch
from paddleocr import PaddleOCR
import time
from fastapi import FastAPI, File, UploadFile, Form
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer
from constant import SYSTEM_PROMPT_VALID, KEYWORD_MODEL

logger = setup_logging(__name__)

from keyword_extracter import extract_keywords

app = FastAPI(docs_url="/form-validation-poc/docs",          # Swagger UI path
    openapi_url="/form-validation-poc/openapi.json" # OpenAPI JSON path
)

# Initialize model and tokenizer
model = OVModelForCausalLM.from_pretrained(KEYWORD_MODEL, device="CPU")
tokenizer = AutoTokenizer.from_pretrained(KEYWORD_MODEL)

def calculate_similarity(text1, text2):
    """Calculate similarity between two strings (0-1)"""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def process_image(image_path, settings):
    """Main processing function"""
    ocr = PaddleOCR(use_angle_cls=True, lang=settings.get("language", "en"))
    result = ocr.ocr(image_path, cls=True)
    return process_dynamic_sections(result, settings, image_path)

def extend_bbox(bbox, extend_x, extend_y):
    """Extend bounding box in all directions"""
    x_min = min(p[0] for p in bbox) - extend_x
    y_min = min(p[1] for p in bbox) - extend_y
    x_max = max(p[0] for p in bbox) + extend_x
    y_max = max(p[1] for p in bbox) + extend_y
    return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]

def is_inside_bbox(word_bbox, target_bbox):
    """Check if word is inside target bounding box"""
    word_x_min = min(p[0] for p in word_bbox)
    word_y_min = min(p[1] for p in word_bbox)
    word_x_max = max(p[0] for p in word_bbox)
    word_y_max = max(p[1] for p in word_bbox)

    target_x_min = min(p[0] for p in target_bbox)
    target_y_min = min(p[1] for p in target_bbox)
    target_x_max = max(p[0] for p in target_bbox)
    target_y_max = max(p[1] for p in target_bbox)

    return (
        target_x_min <= word_x_min <= target_x_max
        and target_x_min <= word_x_max <= target_x_max
        and target_y_min <= word_y_min <= target_y_max
        and target_y_min <= word_y_max <= target_y_max
    )

def process_dynamic_sections(ocr_result, settings, image_path):
    """Match LLM sections with OCR coordinates and perform second OCR on extended BBox."""
    keywords = settings.get("keywords")
    similarity_threshold = settings.get("similarity_threshold")
    best_matches_bboxes_ext = []
    best_matches_text_ext = {}

    for line in ocr_result:
        for word_info in line:
            text = word_info[1][0]
            coordinates = word_info[0]

            for keyword in keywords:
                keyword = str(keyword).replace("'", "")
                similarity = calculate_similarity(text, keyword)   
                if similarity >= similarity_threshold:
                    extended_box = extend_bbox(
                        coordinates,
                        settings["general_extend_x"],
                        settings["general_extend_y"],
                    )
        
                    best_matches_bboxes_ext.append((keyword, extended_box))

                
                    if keyword not in best_matches_text_ext:
                        best_matches_text_ext[keyword] = []
                    break  # Stop checking further keywords for this text

    for line in ocr_result:
        for word_info in line:
            word_text = word_info[1][0]  # Extracted word
            word_coordinates = word_info[0]  # Bounding box

            for tag, extended_box in best_matches_bboxes_ext:
                if is_inside_bbox(word_coordinates, extended_box):
                    best_matches_text_ext[tag].append(word_text)
                    

    return best_matches_text_ext

def openvino_response(prompt, result, temperature=0.2):
    system_prompt = SYSTEM_PROMPT_VALID.format(prompt=prompt)

    # Format prompt with Mistral's instruction tokens
    input_text = f"<s>[INST] {system_prompt}\n\nUser Input: {result} [/INST]"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)

    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=temperature,
            top_p=0.1
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the validation part
        validation_part = response.split("[/INST]")[-1].strip()
        return validation_part
    except Exception as e:
        return f"Error: {str(e)}"
    
@app.post("/process")
async def process_ocr(image: UploadFile = File(...), instruction: str = Form(...)):
    image_bytes = await image.read()
    start_time_full = time.time()
    settings = {
        "language": "en",
        "keywords": extract_keywords(sentence=instruction).split(","),
        "general_extend_x": 90,
        "general_extend_y": 90,
        "similarity_threshold": 0.70,
    }
    print(settings.get("keywords"))
    results = process_image(image_bytes, settings)
    text_result = " ".join([" ".join(texts) for texts in results.values()])
    print(text_result)
    start_time = time.time()
    response = openvino_response(instruction, text_result)
    end_time = time.time()
    response_time = round(end_time - start_time, 2)
    end_time_full = time.time()
    full_response = round(end_time_full - start_time_full, 2)
    return {"Instruction Response": response, "Response Time": full_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app=app, host='0.0.0.0', port=8085)