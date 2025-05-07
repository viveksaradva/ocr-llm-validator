# dynamic_section_api_llamacpp.py
"""
Refactored FastAPI app using TheBloke's quantized model via llama.cpp (`llama-run`) for both
keyword extraction and validation. Heavy objects are initialized once at startup.
Includes benchmarking of LLaMa calls and bounding-box extraction diagnostics.
"""
import os
import subprocess
import time
from difflib import SequenceMatcher
from fastapi import FastAPI, File, UploadFile, Form
from paddleocr import PaddleOCR
from logging_config import setup_logging
from constant import KEYWORDS_EXTRACTOR_PROMPT, SYSTEM_PROMPT_VALID
from PIL import Image, ImageDraw

# --- Setup logging ---
logger = setup_logging(__name__)

# --- FastAPI app ---
app = FastAPI(
    docs_url="/form-validation-poc/docs",
    openapi_url="/form-validation-poc/openapi.json"
)

# --- One-time initialization ---
LLAMA_RUN_BIN = os.getenv("LLAMA_RUN_BIN", "llama.cpp/build/bin/llama-run")
MODEL_PATH = os.getenv(
    "LLAMA_MODEL_PATH",
    "llama.cpp/models/Llama-2-7b-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf"
)
CPU_THREADS = os.cpu_count() or 1

# Initialize OCR
logger.info("Initializing PaddleOCR...")
ocr = PaddleOCR(use_angle_cls=True, lang="en")
logger.info("Using llama.cpp at %s with model %s", LLAMA_RUN_BIN, MODEL_PATH)

# --- Helper Functions ---

def llamacpp_response(system_prompt: str, user_input: str, temperature: float = 0.2) -> str:
    full_prompt = f"<s>[INST] {system_prompt}\n\nUser Input: {user_input} [/INST]"
    logger.debug("Prompt to llama.cpp:\n%s", full_prompt)
    cmd = [LLAMA_RUN_BIN, "--threads", str(CPU_THREADS), "--temp", str(temperature), MODEL_PATH, full_prompt]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = res.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error("llama.cpp error: %s", e.stderr)
        return ""
    if "[/INST]" in output:
        return output.split("[/INST]")[-1].strip()
    return output


def calculate_similarity(text1: str, text2: str) -> float:
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def extend_bbox(bbox, dx, dy):
    x_coords, y_coords = zip(*bbox)
    return [
        [min(x_coords) - dx, min(y_coords) - dy],
        [max(x_coords) + dx, min(y_coords) - dy],
        [max(x_coords) + dx, max(y_coords) + dy],
        [min(x_coords) - dx, max(y_coords) + dy]
    ]


def is_inside_bbox(word_bbox, target_bbox) -> bool:
    xw = [p[0] for p in word_bbox]
    yw = [p[1] for p in word_bbox]
    xt = [p[0] for p in target_bbox]
    yt = [p[1] for p in target_bbox]
    return min(xt) <= min(xw) and max(xw) <= max(xt) and min(yt) <= min(yw) and max(yw) <= max(yt)


def process_dynamic_sections(ocr_result, settings, debug_image_path=None):
    keywords = settings["keywords"]
    thresh = settings["similarity_threshold"]
    bboxes = []
    text_map = {kw: [] for kw in keywords}

    # locate keywords and extended boxes
    for line in ocr_result:
        for word_info in line:
            txt, coords = word_info[1][0], word_info[0]
            for kw in keywords:
                if calculate_similarity(txt, kw) >= thresh:
                    eb = extend_bbox(coords, settings["general_extend_x"], settings["general_extend_y"])
                    bboxes.append((kw, coords, eb))
                    break

    # Debug: visualize bleed boxes
    if debug_image_path:
        img = Image.open(debug_image_path)
        draw = ImageDraw.Draw(img)
        for kw, orig, ext in bboxes:
            # draw original box in red
            draw.line(orig + [orig[0]], fill="red", width=2)
            # draw extended box in blue
            draw.line(ext + [ext[0]], fill="blue", width=2)
        debug_out = debug_image_path.replace('.png', '_bboxes.png')
        img.save(debug_out)
        logger.info("Saved bounding-box debug image to %s", debug_out)

    # collect words inside extended boxes
    for line in ocr_result:
        for word_info in line:
            wtxt, wcoords = word_info[1][0], word_info[0]
            for kw, _, eb in bboxes:
                if is_inside_bbox(wcoords, eb):
                    text_map[kw].append(wtxt)
    return text_map

# --- Endpoint ---

@app.post("/process")
async def process_ocr(image: UploadFile = File(...), instruction: str = Form(...)):
    img_bytes = await image.read()
    tmp = "/tmp/upload.png"
    with open(tmp, "wb") as f:
        f.write(img_bytes)

    start_all = time.time()

    # 1. Keyword extraction
    kw_system = KEYWORDS_EXTRACTOR_PROMPT
    kw_input = f"Input: {instruction}"
    t0 = time.time()
    kw_output = llamacpp_response(kw_system, kw_input)
    t1 = time.time()
    kw_time = round(t1 - t0, 2)
    keywords = [k.strip(' "') for k in kw_output.split(',') if k]

    # 2. OCR
    ocr_res = ocr.ocr(tmp, cls=True)

    # 3. Flatten all OCR-recognized text
    merged_text = " ".join([
        word_info[1][0]
        for line in ocr_res
        for word_info in line
    ])

    # 4. Validation
    val_system = SYSTEM_PROMPT_VALID.format(prompt=instruction, input=merged_text)
    t2 = time.time()
    validation = llamacpp_response(val_system, f"Text: {merged_text}")
    t3 = time.time()
    val_time = round(t3 - t2, 2)

    total = round(time.time() - start_all, 2)
    return {
        "Instruction Response": validation,
        "Keyword Extraction Time": kw_time,
        "Validation Time": val_time,
        "Total Time": total
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("test:app", host='0.0.0.0', port=8085, reload=True)
