import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

app = FastAPI()

# Local path to the downloaded model
model_dir = "./llama-2-13b-chat-hf"
offload_folder = "./offload"  # Folder to store offloaded model parts

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 150

@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt")
        outputs = model.generate(inputs.input_ids.to(device), max_length=request.max_tokens)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated text for prompt '{request.prompt}'")
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating text: {e}")

    return {"generated_text": generated_text}

if __name__ == "__main__":
    from transformers import logging as transformers_logging
    transformers_logging.set_verbosity_info()
    import uvicorn

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_auth_token=True)

    logger.info("Initializing model...")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(model_dir, use_auth_token=True)

    logger.info("Loading model checkpoint...")
    model = load_checkpoint_and_dispatch(
        model,
        model_dir,
        device_map="balanced",  # Automatically balance the layers across CPU and GPU
        offload_folder=offload_folder,
    )

    device = next(model.parameters()).device
    logger.info(f"Model and tokenizer loaded successfully. Model is using device: {device}")

    logger.info("Starting server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
