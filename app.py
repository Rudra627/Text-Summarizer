from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re 
import os
import logging
from fastapi.templating import Jinja2Templates # UI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

# Get model path
model_path = os.path.join(os.path.dirname(__file__), "save_model")
logger.info(f"Loading model from: {model_path}")
logger.info(f"Model path exists: {os.path.exists(model_path)}")

try:
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    # Use standard google/flan-t5-base tokenizer since tokenizer files are not in save_model
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    logger.info("Model and tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("Using MPS device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("Using CUDA device")
else:
    device = torch.device("cpu")
    logger.info("Using CPU device")

model.to(device)

templates = Jinja2Templates(directory=".")

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text) # lines
    text = re.sub(r"\s+", " ", text) # spaces
    text = re.sub(r"<.*?>", " ", text) # html tags <p> <h1>
    text = text.strip().lower()
    return text

def summarize_dialogue(dialogue : str) -> str:
    try:
        dialogue = clean_data(dialogue) # clean
        logger.info(f"Cleaned text: {dialogue[:100]}...")
        
        if not dialogue or len(dialogue.split()) < 5:
            return "Text is too short to summarize. Please provide longer content."

        # Add the required prefix for T5 summarization
        dialogue = "summarize: " + dialogue
        logger.info(f"Text with prefix: {dialogue[:100]}...")

        # tokenize
        inputs = tokenizer(
            dialogue,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        logger.info("Tokenization done")

        # generate the summary => token ids
        with torch.no_grad():
            targets = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=150,
                min_length=10,
                num_beams=4,
                early_stopping=True,
                temperature=0.7,
                do_sample=False,
                decoder_start_token_id=model.config.decoder_start_token_id
            )
        
        logger.info("Generation done")
        
        # decoded our output
        summary = tokenizer.decode(targets[0], skip_special_tokens=True) # EOS, SEP
        logger.info(f"Summary generated: {summary}")
        return summary
    except Exception as e:
        logger.error(f"Error in summarize_dialogue: {e}")
        raise


@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    try:
        logger.info(f"Received summarize request with text length: {len(dialogue_input.dialogue)}")
        summary = summarize_dialogue(dialogue_input.dialogue)
        logger.info(f"Returning summary: {summary}")
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Error in summarize endpoint: {e}")
        return {"summary": f"Error: {str(e)}", "error": True}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(name="index.html", context={}, request=request)