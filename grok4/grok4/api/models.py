"""
Grok4 API Models

Pydantic models for API request and response schemas.
"""

from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Text generation request."""
    prompt: str = Field(..., description="Input prompt for text generation")
    max_tokens: int = Field(100, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")
    top_k: int = Field(40, ge=0, description="Top-k sampling")
    do_sample: bool = Field(True, description="Whether to use sampling")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")
    stop_tokens: Optional[List[str]] = Field(None, description="Stop tokens")
    stream: bool = Field(False, description="Whether to stream the response")


class GenerateResponse(BaseModel):
    """Text generation response."""
    id: str = Field(..., description="Unique identifier for the generation")
    object: str = Field("text_completion", description="Object type")
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model used for generation")
    choices: List[Dict[str, Any]] = Field(..., description="Generated choices")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")
    generation_time: float = Field(..., description="Time taken for generation")


class ChatMessage(BaseModel):
    """Chat message."""
    role: str = Field(..., description="Message role (system, user, assistant)")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat completion request."""
    messages: List[ChatMessage] = Field(..., description="List of messages")
    max_tokens: int = Field(100, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling")
    stream: bool = Field(False, description="Whether to stream the response")


class ChatResponse(BaseModel):
    """Chat completion response."""
    id: str = Field(..., description="Unique identifier")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model used")
    choices: List[Dict[str, Any]] = Field(..., description="Chat choices")
    usage: Dict[str, int] = Field(..., description="Token usage")
    generation_time: float = Field(..., description="Generation time")


class EmbeddingRequest(BaseModel):
    """Embedding request."""
    input: Union[str, List[str]] = Field(..., description="Text(s) to embed")
    model: str = Field("grok4-base", description="Model to use")


class EmbeddingResponse(BaseModel):
    """Embedding response."""
    object: str = Field("list", description="Object type")
    data: List[Dict[str, Any]] = Field(..., description="Embedding data")
    model: str = Field(..., description="Model used")
    usage: Dict[str, int] = Field(..., description="Token usage")
    generation_time: float = Field(..., description="Generation time")


class ModelInfo(BaseModel):
    """Model information."""
    id: str = Field(..., description="Model identifier")
    object: str = Field("model", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    owned_by: str = Field(..., description="Model owner")
    permission: List[Dict[str, Any]] = Field(default_factory=list, description="Permissions")
    root: str = Field(..., description="Root model")
    parent: Optional[str] = Field(None, description="Parent model")


class ModelsResponse(BaseModel):
    """Models list response."""
    object: str = Field("list", description="Object type")
    data: List[ModelInfo] = Field(..., description="List of models")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: float = Field(..., description="Timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device being used")


class ErrorResponse(BaseModel):
    """Error response."""
    error: Dict[str, Any] = Field(..., description="Error details")
    timestamp: float = Field(..., description="Error timestamp")