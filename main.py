from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ner.registry import ModelRegistry

app = FastAPI(title="NER Serving API", version="1.0.0")

registry = ModelRegistry("config.yaml")


class NERRequest(BaseModel):
    text: str = Field(
        "The F-16 fighter jet was sold to Belgium by Lockheed Martin.",
        example="The F-16 fighter jet was sold to Belgium by Lockheed Martin.",
    )
    model_name: str = Field(
        "gliner_multi-v2.1",
        example="gliner_multi-v2.1",
        description="Select which model to use for NER",
    )


@app.get("/models")
def list_models():
    """List available models and their metadata."""
    return registry.list_models()


@app.post("/ner")
def run_ner(request: NERRequest):
    """Run NER inference using the selected model."""
    try:
        model = registry.get_model(request.model_name)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_name}' not found."
        )
    results = model.predict(request.text)
    return {"model": request.model_name, "entities": results}


@app.get("/healthz")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "loaded_models": list(registry.models.keys())}
