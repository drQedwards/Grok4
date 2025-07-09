"""
Grok4 API Server

FastAPI server for serving Grok4 models with OpenAI-compatible endpoints.
"""

import asyncio
import time
import uuid
from typing import List, Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
from pydantic import BaseModel
import json

from ..models.grok4_model import Grok4Model
from ..utils.config import Config
from ..utils.logger import get_logger, setup_logging
from .models import (
    GenerateRequest, GenerateResponse,
    ChatRequest, ChatResponse, ChatMessage,
    EmbeddingRequest, EmbeddingResponse,
    ModelInfo, ModelsResponse,
    HealthResponse
)

logger = get_logger(__name__)

# Global model instance
_model: Optional[Grok4Model] = None
_tokenizer = None
_config: Optional[Config] = None


async def get_model() -> Grok4Model:
    """Get the global model instance."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _model


async def get_tokenizer():
    """Get the global tokenizer instance."""
    if _tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded")
    return _tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _model, _tokenizer, _config
    
    # Startup
    logger.info("Starting Grok4 API server...")
    
    # Load model and tokenizer
    try:
        # Default configuration
        _config = Config()
        
        # Initialize model
        _model = Grok4Model(config=_config)
        logger.info(f"Loaded Grok4 model with {_model.num_parameters():,} parameters")
        
        # Load tokenizer
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained("gpt2")
        _tokenizer.pad_token = _tokenizer.eos_token
        logger.info("Loaded tokenizer")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            _model = _model.cuda()
            logger.info(f"Model moved to GPU: {torch.cuda.get_device_name()}")
        
        logger.info("Grok4 API server startup complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Grok4 API server...")


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Grok4 API",
        description="Advanced AI Framework for Language Understanding and Generation",
        version="0.1.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


@app.get("/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        model_loaded=_model is not None,
        device="cuda" if torch.cuda.is_available() and _model is not None else "cpu"
    )


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models."""
    models = [
        ModelInfo(
            id="grok4-base",
            object="model",
            created=int(time.time()),
            owned_by="grok4",
            permission=[],
            root="grok4-base",
            parent=None
        )
    ]
    
    return ModelsResponse(
        object="list",
        data=models
    )


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate_text(
    request: GenerateRequest,
    model: Grok4Model = Depends(get_model),
    tokenizer = Depends(get_tokenizer)
):
    """Generate text from a prompt."""
    try:
        start_time = time.time()
        
        # Generate text
        generated_text = model.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            do_sample=request.do_sample,
            repetition_penalty=request.repetition_penalty,
            stop_tokens=request.stop_tokens,
            tokenizer=tokenizer
        )
        
        generation_time = time.time() - start_time
        
        return GenerateResponse(
            id=f"gen-{uuid.uuid4()}",
            object="text_completion",
            created=int(time.time()),
            model="grok4-base",
            choices=[{
                "text": generated_text,
                "index": 0,
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(tokenizer.encode(request.prompt)),
                "completion_tokens": len(tokenizer.encode(generated_text)),
                "total_tokens": len(tokenizer.encode(request.prompt + generated_text))
            },
            generation_time=generation_time
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completion(
    request: ChatRequest,
    model: Grok4Model = Depends(get_model),
    tokenizer = Depends(get_tokenizer)
):
    """Chat completion endpoint."""
    try:
        start_time = time.time()
        
        # Convert messages to the expected format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Generate response
        response = model.chat(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            tokenizer=tokenizer
        )
        
        generation_time = time.time() - start_time
        
        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model="grok4-base",
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": sum(len(tokenizer.encode(msg["content"])) for msg in messages),
                "completion_tokens": len(tokenizer.encode(response)),
                "total_tokens": sum(len(tokenizer.encode(msg["content"])) for msg in messages) + len(tokenizer.encode(response))
            },
            generation_time=generation_time
        )
        
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    model: Grok4Model = Depends(get_model),
    tokenizer = Depends(get_tokenizer)
):
    """Create embeddings for input texts."""
    try:
        start_time = time.time()
        
        # Ensure input is a list
        texts = request.input if isinstance(request.input, list) else [request.input]
        
        # Generate embeddings
        embeddings = model.embed(texts, tokenizer=tokenizer)
        embeddings = embeddings.cpu().numpy().tolist()
        
        generation_time = time.time() - start_time
        
        # Create embedding objects
        embedding_objects = []
        for i, embedding in enumerate(embeddings):
            embedding_objects.append({
                "object": "embedding",
                "embedding": embedding,
                "index": i
            })
        
        total_tokens = sum(len(tokenizer.encode(text)) for text in texts)
        
        return EmbeddingResponse(
            object="list",
            data=embedding_objects,
            model="grok4-base",
            usage={
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            },
            generation_time=generation_time
        )
        
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_generate(
    prompt: str,
    model: Grok4Model,
    tokenizer,
    max_tokens: int = 100,
    temperature: float = 0.7,
    **kwargs
) -> AsyncGenerator[str, None]:
    """Stream text generation."""
    # This is a simplified streaming implementation
    # In a real implementation, you'd modify the model's generate method
    # to yield tokens as they're generated
    
    generated_text = model.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        tokenizer=tokenizer,
        **kwargs
    )
    
    # Simulate streaming by yielding words
    words = generated_text.split()
    for i, word in enumerate(words):
        chunk = {
            "id": f"gen-{uuid.uuid4()}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "grok4-base",
            "choices": [{
                "text": word + (" " if i < len(words) - 1 else ""),
                "index": 0,
                "finish_reason": None if i < len(words) - 1 else "stop"
            }]
        }
        
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.05)  # Simulate streaming delay
    
    yield "data: [DONE]\n\n"


@app.post("/v1/generate/stream")
async def stream_generate_text(
    request: GenerateRequest,
    model: Grok4Model = Depends(get_model),
    tokenizer = Depends(get_tokenizer)
):
    """Stream text generation."""
    try:
        return StreamingResponse(
            stream_generate(
                prompt=request.prompt,
                model=model,
                tokenizer=tokenizer,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                do_sample=request.do_sample,
                repetition_penalty=request.repetition_penalty,
                stop_tokens=request.stop_tokens,
            ),
            media_type="text/plain",
            headers={"Content-Type": "text/event-stream"}
        )
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Main entry point for CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Grok4 API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model-path", help="Path to model checkpoint")
    parser.add_argument("--config-path", help="Path to configuration file")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Load custom model if specified
    global _model, _config
    if args.model_path:
        logger.info(f"Loading model from {args.model_path}")
        _model = Grok4Model.from_pretrained(args.model_path)
    
    if args.config_path:
        logger.info(f"Loading config from {args.config_path}")
        _config = Config.from_file(args.config_path)
    
    # Start server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower()
    )


if __name__ == "__main__":
    main()