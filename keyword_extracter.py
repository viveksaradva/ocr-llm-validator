from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM
from constant import KEYWORD_MODEL, KEYWORDS_EXTRACTOR_PROMPT
from dotenv import load_dotenv
import os
from logging_config import setup_logging
logger = setup_logging(__name__)

load_dotenv()

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(KEYWORD_MODEL)
model = OVModelForCausalLM.from_pretrained(KEYWORD_MODEL, device="CPU")

def generate_response(sentence, temperature=0.2):
    logger.info(f"Keyword Extraction Start!")
    system_prompt = KEYWORDS_EXTRACTOR_PROMPT
    
    # Format prompt with Mistral's instruction tokens
    input_text = f"<s>[INST] {system_prompt}\n\nUser Input: {sentence} [/INST]"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)

    try:
        # Generate response with OpenVINO
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=temperature,
            top_p=0.1
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the keyword part (remove prompt and input)
        keyword_part = response.split("[/INST]")[-1].strip()
        logger.info("Keyword Extraction End!")
        return keyword_part
    except Exception as e:
        return f"Error: {str(e)}"

def extract_keywords(sentence):
    return generate_response(sentence=sentence)

    # Load a lightweight model
    
    # # Load a lightweight model
    # generator = pipeline(
    #     "text-generation", 
    #     model=KEYWORD_MODEL,
    #     tokenizer=KEYWORD_TOKENIZER,
    #     device="cpu",
    #     token=TOKEN
    # )

    # response = generator(
    #     messages, 
    #     max_new_tokens=200, 
    #     do_sample=True, 
    #     truncation=True,  
    #     temperature=0.2,
    #     top_p=0.1
    # )
    # # print(response)
    # return response[0]["generated_text"][2]["content"].strip()

# def extract_keywords(sentence):
#     return generate_response(sentence=sentence)