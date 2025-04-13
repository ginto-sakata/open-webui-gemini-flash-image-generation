import asyncio
import base64
import io
import json
import logging
import mimetypes
import re
from pathlib import Path
from typing import Optional

import requests
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from open_webui.config import CACHE_DIR
from open_webui.constants import ERROR_MESSAGES
from open_webui.env import ENABLE_FORWARD_USER_INFO_HEADERS, SRC_LOG_LEVELS
from open_webui.routers.files import upload_file
from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.images.comfyui import (
    ComfyUIGenerateImageForm,
    ComfyUIWorkflow,
    comfyui_generate_image,
)
from pydantic import BaseModel

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["IMAGES"])

IMAGE_CACHE_DIR = CACHE_DIR / "image" / "generations"
IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


router = APIRouter()


@router.get("/config")
async def get_config(request: Request, user=Depends(get_admin_user)):
    return {
        "enabled": request.app.state.config.ENABLE_IMAGE_GENERATION,
        "engine": request.app.state.config.IMAGE_GENERATION_ENGINE,
        "prompt_generation": request.app.state.config.ENABLE_IMAGE_PROMPT_GENERATION,
        "openai": {
            "OPENAI_API_BASE_URL": request.app.state.config.IMAGES_OPENAI_API_BASE_URL,
            "OPENAI_API_KEY": request.app.state.config.IMAGES_OPENAI_API_KEY,
        },
        "automatic1111": {
            "AUTOMATIC1111_BASE_URL": request.app.state.config.AUTOMATIC1111_BASE_URL,
            "AUTOMATIC1111_API_AUTH": request.app.state.config.AUTOMATIC1111_API_AUTH,
            "AUTOMATIC1111_CFG_SCALE": request.app.state.config.AUTOMATIC1111_CFG_SCALE,
            "AUTOMATIC1111_SAMPLER": request.app.state.config.AUTOMATIC1111_SAMPLER,
            "AUTOMATIC1111_SCHEDULER": request.app.state.config.AUTOMATIC1111_SCHEDULER,
        },
        "comfyui": {
            "COMFYUI_BASE_URL": request.app.state.config.COMFYUI_BASE_URL,
            "COMFYUI_API_KEY": request.app.state.config.COMFYUI_API_KEY,
            "COMFYUI_WORKFLOW": request.app.state.config.COMFYUI_WORKFLOW,
            "COMFYUI_WORKFLOW_NODES": request.app.state.config.COMFYUI_WORKFLOW_NODES,
        },
        "gemini": {
            "GEMINI_API_BASE_URL": request.app.state.config.IMAGES_GEMINI_API_BASE_URL,
            "GEMINI_API_KEY": request.app.state.config.IMAGES_GEMINI_API_KEY,
        },
    }


class OpenAIConfigForm(BaseModel):
    OPENAI_API_BASE_URL: str
    OPENAI_API_KEY: str


class Automatic1111ConfigForm(BaseModel):
    AUTOMATIC1111_BASE_URL: str
    AUTOMATIC1111_API_AUTH: str
    AUTOMATIC1111_CFG_SCALE: Optional[str | float | int]
    AUTOMATIC1111_SAMPLER: Optional[str]
    AUTOMATIC1111_SCHEDULER: Optional[str]


class ComfyUIConfigForm(BaseModel):
    COMFYUI_BASE_URL: str
    COMFYUI_API_KEY: str
    COMFYUI_WORKFLOW: str
    COMFYUI_WORKFLOW_NODES: list[dict]


class GeminiConfigForm(BaseModel):
    GEMINI_API_BASE_URL: str
    GEMINI_API_KEY: str


class ConfigForm(BaseModel):
    enabled: bool
    engine: str
    prompt_generation: bool
    openai: OpenAIConfigForm
    automatic1111: Automatic1111ConfigForm
    comfyui: ComfyUIConfigForm
    gemini: GeminiConfigForm


@router.post("/config/update")
async def update_config(
    request: Request, form_data: ConfigForm, user=Depends(get_admin_user)
):
    request.app.state.config.IMAGE_GENERATION_ENGINE = form_data.engine
    request.app.state.config.ENABLE_IMAGE_GENERATION = form_data.enabled

    request.app.state.config.ENABLE_IMAGE_PROMPT_GENERATION = (
        form_data.prompt_generation
    )

    request.app.state.config.IMAGES_OPENAI_API_BASE_URL = (
        form_data.openai.OPENAI_API_BASE_URL
    )
    request.app.state.config.IMAGES_OPENAI_API_KEY = form_data.openai.OPENAI_API_KEY

    request.app.state.config.IMAGES_GEMINI_API_BASE_URL = (
        form_data.gemini.GEMINI_API_BASE_URL
    )
    request.app.state.config.IMAGES_GEMINI_API_KEY = form_data.gemini.GEMINI_API_KEY

    request.app.state.config.AUTOMATIC1111_BASE_URL = (
        form_data.automatic1111.AUTOMATIC1111_BASE_URL
    )
    request.app.state.config.AUTOMATIC1111_API_AUTH = (
        form_data.automatic1111.AUTOMATIC1111_API_AUTH
    )

    request.app.state.config.AUTOMATIC1111_CFG_SCALE = (
        float(form_data.automatic1111.AUTOMATIC1111_CFG_SCALE)
        if form_data.automatic1111.AUTOMATIC1111_CFG_SCALE
        else None
    )
    request.app.state.config.AUTOMATIC1111_SAMPLER = (
        form_data.automatic1111.AUTOMATIC1111_SAMPLER
        if form_data.automatic1111.AUTOMATIC1111_SAMPLER
        else None
    )
    request.app.state.config.AUTOMATIC1111_SCHEDULER = (
        form_data.automatic1111.AUTOMATIC1111_SCHEDULER
        if form_data.automatic1111.AUTOMATIC1111_SCHEDULER
        else None
    )

    request.app.state.config.COMFYUI_BASE_URL = (
        form_data.comfyui.COMFYUI_BASE_URL.strip("/")
    )
    request.app.state.config.COMFYUI_API_KEY = form_data.comfyui.COMFYUI_API_KEY

    request.app.state.config.COMFYUI_WORKFLOW = form_data.comfyui.COMFYUI_WORKFLOW
    request.app.state.config.COMFYUI_WORKFLOW_NODES = (
        form_data.comfyui.COMFYUI_WORKFLOW_NODES
    )

    return {
        "enabled": request.app.state.config.ENABLE_IMAGE_GENERATION,
        "engine": request.app.state.config.IMAGE_GENERATION_ENGINE,
        "prompt_generation": request.app.state.config.ENABLE_IMAGE_PROMPT_GENERATION,
        "openai": {
            "OPENAI_API_BASE_URL": request.app.state.config.IMAGES_OPENAI_API_BASE_URL,
            "OPENAI_API_KEY": request.app.state.config.IMAGES_OPENAI_API_KEY,
        },
        "automatic1111": {
            "AUTOMATIC1111_BASE_URL": request.app.state.config.AUTOMATIC1111_BASE_URL,
            "AUTOMATIC1111_API_AUTH": request.app.state.config.AUTOMATIC1111_API_AUTH,
            "AUTOMATIC1111_CFG_SCALE": request.app.state.config.AUTOMATIC1111_CFG_SCALE,
            "AUTOMATIC1111_SAMPLER": request.app.state.config.AUTOMATIC1111_SAMPLER,
            "AUTOMATIC1111_SCHEDULER": request.app.state.config.AUTOMATIC1111_SCHEDULER,
        },
        "comfyui": {
            "COMFYUI_BASE_URL": request.app.state.config.COMFYUI_BASE_URL,
            "COMFYUI_API_KEY": request.app.state.config.COMFYUI_API_KEY,
            "COMFYUI_WORKFLOW": request.app.state.config.COMFYUI_WORKFLOW,
            "COMFYUI_WORKFLOW_NODES": request.app.state.config.COMFYUI_WORKFLOW_NODES,
        },
        "gemini": {
            "GEMINI_API_BASE_URL": request.app.state.config.IMAGES_GEMINI_API_BASE_URL,
            "GEMINI_API_KEY": request.app.state.config.IMAGES_GEMINI_API_KEY,
        },
    }


def get_automatic1111_api_auth(request: Request):
    if request.app.state.config.AUTOMATIC1111_API_AUTH is None:
        return ""
    else:
        auth1111_byte_string = request.app.state.config.AUTOMATIC1111_API_AUTH.encode(
            "utf-8"
        )
        auth1111_base64_encoded_bytes = base64.b64encode(auth1111_byte_string)
        auth1111_base64_encoded_string = auth1111_base64_encoded_bytes.decode("utf-8")
        return f"Basic {auth1111_base64_encoded_string}"


@router.get("/config/url/verify")
async def verify_url(request: Request, user=Depends(get_admin_user)):
    if request.app.state.config.IMAGE_GENERATION_ENGINE == "automatic1111":
        try:
            r = requests.get(
                url=f"{request.app.state.config.AUTOMATIC1111_BASE_URL}/sdapi/v1/options",
                headers={"authorization": get_automatic1111_api_auth(request)},
            )
            r.raise_for_status()
            return True
        except Exception:
            request.app.state.config.ENABLE_IMAGE_GENERATION = False
            raise HTTPException(status_code=400, detail=ERROR_MESSAGES.INVALID_URL)
    elif request.app.state.config.IMAGE_GENERATION_ENGINE == "comfyui":

        headers = None
        if request.app.state.config.COMFYUI_API_KEY:
            headers = {
                "Authorization": f"Bearer {request.app.state.config.COMFYUI_API_KEY}"
            }

        try:
            r = requests.get(
                url=f"{request.app.state.config.COMFYUI_BASE_URL}/object_info",
                headers=headers,
            )
            r.raise_for_status()
            return True
        except Exception:
            request.app.state.config.ENABLE_IMAGE_GENERATION = False
            raise HTTPException(status_code=400, detail=ERROR_MESSAGES.INVALID_URL)
    else:
        # Assuming other engines don't need verification or it's handled elsewhere
        return True


def set_image_model(request: Request, model: str):
    log.info(f"Setting image model to {model}")
    request.app.state.config.IMAGE_GENERATION_MODEL = model
    if request.app.state.config.IMAGE_GENERATION_ENGINE in ["", "automatic1111"]:
        try:
            api_auth = get_automatic1111_api_auth(request)
            r = requests.get(
                url=f"{request.app.state.config.AUTOMATIC1111_BASE_URL}/sdapi/v1/options",
                headers={"authorization": api_auth},
            )
            r.raise_for_status() # Check if request was successful
            options = r.json()
            if model != options["sd_model_checkpoint"]:
                options["sd_model_checkpoint"] = model
                r_post = requests.post(
                    url=f"{request.app.state.config.AUTOMATIC1111_BASE_URL}/sdapi/v1/options",
                    json=options,
                    headers={"authorization": api_auth},
                )
                r_post.raise_for_status() # Check if post was successful
        except Exception as e:
            log.error(f"Failed to set Automatic1111 model: {e}")
            # Decide if this should raise an error or just log
            # raise HTTPException(status_code=500, detail=f"Failed to set Automatic1111 model: {e}")

    return request.app.state.config.IMAGE_GENERATION_MODEL


def get_image_model(request):
    if request.app.state.config.IMAGE_GENERATION_ENGINE == "openai":
        return (
            request.app.state.config.IMAGE_GENERATION_MODEL
            if request.app.state.config.IMAGE_GENERATION_MODEL
            else "dall-e-2"
        )
    elif request.app.state.config.IMAGE_GENERATION_ENGINE == "gemini":
        # Ensure a default is returned if nothing is set, using a common recent model
        return (
            request.app.state.config.IMAGE_GENERATION_MODEL
            if request.app.state.config.IMAGE_GENERATION_MODEL
            else "gemini-2.0-flash-exp-image-generation" # Or perhaps a specific image gen model like "imagen-..." if preferred as default
        )
    elif request.app.state.config.IMAGE_GENERATION_ENGINE == "comfyui":
        return (
            request.app.state.config.IMAGE_GENERATION_MODEL
            if request.app.state.config.IMAGE_GENERATION_MODEL
            else ""
        )
    elif (
        request.app.state.config.IMAGE_GENERATION_ENGINE == "automatic1111"
        or request.app.state.config.IMAGE_GENERATION_ENGINE == ""
    ):
        try:
            r = requests.get(
                url=f"{request.app.state.config.AUTOMATIC1111_BASE_URL}/sdapi/v1/options",
                headers={"authorization": get_automatic1111_api_auth(request)},
            )
            r.raise_for_status()
            options = r.json()
            return options["sd_model_checkpoint"]
        except Exception as e:
            log.error(f"Could not fetch Automatic1111 model: {e}. Disabling image generation.")
            request.app.state.config.ENABLE_IMAGE_GENERATION = False
            # Returning None or empty string might be better than raising here
            return None
            # raise HTTPException(status_code=400, detail=ERROR_MESSAGES.DEFAULT(e))


class ImageConfigForm(BaseModel):
    MODEL: str
    IMAGE_SIZE: str
    IMAGE_STEPS: int


@router.get("/image/config")
async def get_image_config(request: Request, user=Depends(get_admin_user)):
    return {
        "MODEL": request.app.state.config.IMAGE_GENERATION_MODEL,
        "IMAGE_SIZE": request.app.state.config.IMAGE_SIZE,
        "IMAGE_STEPS": request.app.state.config.IMAGE_STEPS,
    }


@router.post("/image/config/update")
async def update_image_config(
    request: Request, form_data: ImageConfigForm, user=Depends(get_admin_user)
):
    # Assuming set_image_model handles potential errors or logs them
    set_image_model(request, form_data.MODEL)

    pattern = r"^\d+x\d+$"
    if re.match(pattern, form_data.IMAGE_SIZE):
        request.app.state.config.IMAGE_SIZE = form_data.IMAGE_SIZE
    else:
        raise HTTPException(
            status_code=400,
            detail=ERROR_MESSAGES.INCORRECT_FORMAT("Image size must be WxH (e.g., 512x512)."),
        )

    # Ensure steps is an integer and non-negative
    try:
        steps = int(form_data.IMAGE_STEPS)
        if steps >= 0:
            request.app.state.config.IMAGE_STEPS = steps
        else:
            raise ValueError("Steps must be non-negative.")
    except (ValueError, TypeError):
         raise HTTPException(
            status_code=400,
            detail=ERROR_MESSAGES.INCORRECT_FORMAT("Image steps must be a non-negative integer (e.g., 50)."),
        )


    return {
        "MODEL": request.app.state.config.IMAGE_GENERATION_MODEL,
        "IMAGE_SIZE": request.app.state.config.IMAGE_SIZE,
        "IMAGE_STEPS": request.app.state.config.IMAGE_STEPS,
    }


@router.get("/models")
def get_models(request: Request, user=Depends(get_verified_user)):
    try:
        if request.app.state.config.IMAGE_GENERATION_ENGINE == "openai":
            return [
                {"id": "dall-e-2", "name": "DALL·E 2"},
                {"id": "dall-e-3", "name": "DALL·E 3"},
            ]
        elif request.app.state.config.IMAGE_GENERATION_ENGINE == "gemini":
             # Provide a list of known/common Gemini models capable of image generation
             # This might need updating as Google releases/retires models
            return [
                {"id": "gemini-2.0-flash-exp-image-generation", "name": "Gemini 2.0 Flash (Image Generation) Experimental"}
            ]
        elif request.app.state.config.IMAGE_GENERATION_ENGINE == "comfyui":
            # TODO - get models from comfyui (Original code)
            headers = None # Initialize headers
            if request.app.state.config.COMFYUI_API_KEY:
                headers = {
                    "Authorization": f"Bearer {request.app.state.config.COMFYUI_API_KEY}"
                }
            r = requests.get(
                url=f"{request.app.state.config.COMFYUI_BASE_URL}/object_info",
                headers=headers, # Pass headers possibly containing auth
            )
            r.raise_for_status() # Check for request errors
            info = r.json()

            # Safely parse workflow JSON
            workflow = None
            try:
                if request.app.state.config.COMFYUI_WORKFLOW:
                     workflow = json.loads(request.app.state.config.COMFYUI_WORKFLOW)
            except json.JSONDecodeError:
                 log.error("Failed to parse ComfyUI workflow JSON.")
                 workflow = None # Ensure workflow is None if parsing fails


            model_node_id = None
            # Check if workflow and nodes config exist before iterating
            if workflow and request.app.state.config.COMFYUI_WORKFLOW_NODES:
                for node in request.app.state.config.COMFYUI_WORKFLOW_NODES:
                    if node.get("type") == "model" and node.get("node_ids"):
                        model_node_id = node["node_ids"][0]
                        break

            if model_node_id and workflow and model_node_id in workflow:
                model_list_key = None
                node_info = workflow[model_node_id]
                class_type = node_info.get("class_type")

                if class_type and class_type in info and "input" in info[class_type] and "required" in info[class_type]["input"]:
                    required_inputs = info[class_type]["input"]["required"]
                    for key in required_inputs:
                        if "_name" in key and isinstance(required_inputs[key], list) and required_inputs[key]:
                            model_list_key = key
                            break

                    if model_list_key:
                        model_list = required_inputs[model_list_key][0]
                        if isinstance(model_list, list): # Ensure it's a list of models
                             return [{"id": model, "name": model} for model in model_list]


            # Fallback or default logic if specific node logic fails
            # Check if 'CheckpointLoaderSimple' exists and has the expected structure
            if "CheckpointLoaderSimple" in info and \
               "input" in info["CheckpointLoaderSimple"] and \
               "required" in info["CheckpointLoaderSimple"]["input"] and \
               "ckpt_name" in info["CheckpointLoaderSimple"]["input"]["required"] and \
               isinstance(info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"], list) and \
               info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"]:

                ckpt_list = info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
                if isinstance(ckpt_list, list): # Final check it's a list
                    return [{"id": model, "name": model} for model in ckpt_list]

            # If all attempts fail, return empty list or log error
            log.warning("Could not determine ComfyUI model list from workflow or default.")
            return []

        elif (
            request.app.state.config.IMAGE_GENERATION_ENGINE == "automatic1111"
            or request.app.state.config.IMAGE_GENERATION_ENGINE == ""
        ):
            r = requests.get(
                url=f"{request.app.state.config.AUTOMATIC1111_BASE_URL}/sdapi/v1/sd-models",
                headers={"authorization": get_automatic1111_api_auth(request)},
            )
            r.raise_for_status() # Check for request errors
            models = r.json()
            # Ensure models is a list and items are dicts with required keys
            if isinstance(models, list):
                 return [
                    {"id": model["title"], "name": model["model_name"]}
                    for model in models if isinstance(model, dict) and "title" in model and "model_name" in model
                 ]
            else:
                 log.error("Unexpected response format from Automatic1111 /sd-models endpoint.")
                 return []
        else:
            # Handle unknown engine type if necessary
             log.warning(f"Unknown image generation engine configured: {request.app.state.config.IMAGE_GENERATION_ENGINE}")
             return []

    except Exception as e:
        log.error(f"Failed to get image models list: {e}", exc_info=True)
        # Avoid disabling generation on list failure, just return empty
        # request.app.state.config.ENABLE_IMAGE_GENERATION = False
        # Don't raise HTTPException, just return empty list for graceful UI handling
        return []
        # raise HTTPException(status_code=400, detail=ERROR_MESSAGES.DEFAULT(e))


class GenerateImageForm(BaseModel):
    model: Optional[str] = None
    prompt: str
    size: Optional[str] = None
    n: int = 1
    negative_prompt: Optional[str] = None


def load_b64_image_data(b64_str):
    try:
        # Handle potential data URI prefix
        if isinstance(b64_str, str) and "," in b64_str:
            header, encoded = b64_str.split(",", 1)
            # Extract mime type if available, default otherwise
            if header.startswith("data:") and ";base64" in header:
                 mime_type = header.split(":")[1].split(";")[0]
            else:
                 mime_type = "image/png" # Default if header format is unexpected
            img_data = base64.b64decode(encoded)
        # Handle raw base64 string
        elif isinstance(b64_str, str):
            mime_type = "image/png" # Assume PNG if no header
            img_data = base64.b64decode(b64_str)
        else:
             log.error("Invalid input type for base64 decoding.")
             return None, None # Return two values

        return img_data, mime_type
    except Exception as e:
        log.exception(f"Error loading base64 image data: {e}")
        return None, None # Return two values on error


def load_url_image_data(url, headers=None):
    try:
        # Use provided headers if available
        response = requests.get(url, headers=headers, stream=True, timeout=30) # Add timeout
        response.raise_for_status() # Check for HTTP errors

        content_type = response.headers.get("content-type")
        if content_type and content_type.startswith("image/"):
            # Read the content now that we know it's likely an image
            img_data = response.content
            return img_data, content_type
        else:
            log.error(f"URL content type ('{content_type}') is not an image.")
            return None, None # Return two values

    except requests.exceptions.RequestException as e:
        log.exception(f"Error fetching image from URL '{url}': {e}")
        return None, None # Return two values on error


def upload_image(request, image_metadata, image_data, content_type, user):
    # Ensure content_type is valid before guessing extension
    if not content_type or not isinstance(content_type, str):
        log.warning(f"Invalid content_type '{content_type}', defaulting extension to .png")
        image_format = ".png"
    else:
        image_format = mimetypes.guess_extension(content_type)
        if not image_format:
            log.warning(f"Could not guess extension for content_type '{content_type}', defaulting to .png")
            image_format = ".png" # Default if guess fails


    # Ensure image_data is bytes
    if not isinstance(image_data, bytes):
        log.error("Image data provided to upload_image is not bytes.")
        return None # Cannot upload non-bytes data

    try:
        # Use io.BytesIO for the file-like object
        file_obj = io.BytesIO(image_data)
        upload_file_obj = UploadFile(
            file=file_obj,
            filename=f"generated-image{image_format}",
            headers={"content-type": content_type if content_type else "application/octet-stream"}
        )

        # Call the file upload router function
        # Ensure 'process=False' if available in upload_file signature and desired
        # Assuming upload_file returns a model with an 'id' attribute
        file_item = upload_file(
             request,
             file=upload_file_obj,
             user=user,
             file_metadata=image_metadata,
             process=False # Explicitly skip RAG processing for generated images
        )

        if file_item and hasattr(file_item, 'id'):
            # Construct the URL for accessing the uploaded file
            # Ensure the route name 'get_file_content_by_id' is correct
            url = request.url_for("get_file_content_by_id", id=file_item.id)
            return str(url) # Return the URL as a string
        else:
            log.error("File upload did not return a valid item with an ID.")
            return None

    except Exception as e:
        log.exception(f"Error during image upload process: {e}")
        return None


@router.post("/generations")
async def image_generations(
    request: Request,
    form_data: GenerateImageForm,
    user=Depends(get_verified_user),
):
    # Validate and parse image size safely
    try:
        width, height = tuple(map(int, request.app.state.config.IMAGE_SIZE.split("x")))
    except (ValueError, AttributeError):
        log.error(f"Invalid IMAGE_SIZE format: {request.app.state.config.IMAGE_SIZE}. Using default 512x512.")
        width, height = 512, 512 # Default size


    r = None # Initialize response variable
    try:
        if request.app.state.config.IMAGE_GENERATION_ENGINE == "openai":
            headers = {}
            api_key = request.app.state.config.IMAGES_OPENAI_API_KEY
            base_url = request.app.state.config.IMAGES_OPENAI_API_BASE_URL

            if not api_key or not base_url:
                 raise HTTPException(status_code=500, detail="OpenAI API Key or Base URL not configured.")

            headers["Authorization"] = f"Bearer {api_key}"
            headers["Content-Type"] = "application/json"

            if ENABLE_FORWARD_USER_INFO_HEADERS:
                headers["X-OpenWebUI-User-Name"] = user.name
                headers["X-OpenWebUI-User-Id"] = user.id
                headers["X-OpenWebUI-User-Email"] = user.email
                headers["X-OpenWebUI-User-Role"] = user.role

            # Determine model, using default if necessary
            model = form_data.model or request.app.state.config.IMAGE_GENERATION_MODEL or "dall-e-2"

            data = {
                "model": model,
                "prompt": form_data.prompt,
                "n": form_data.n if form_data.n > 0 else 1, # Ensure n is positive
                "size": form_data.size or request.app.state.config.IMAGE_SIZE,
                "response_format": "b64_json", # Request base64
            }

            api_url = f"{base_url.strip('/')}/images/generations"

            r = await asyncio.to_thread(requests.post, url=api_url, json=data, headers=headers, timeout=180) # Add timeout
            r.raise_for_status()
            res = r.json()

            images = []
            if "data" in res and isinstance(res["data"], list):
                for image_item in res["data"]:
                    image_data = None
                    content_type = None
                    # Prefer b64_json if available
                    if "b64_json" in image_item:
                        image_data, content_type = load_b64_image_data(image_item["b64_json"])
                    # Fallback to URL if b64 not present
                    elif "url" in image_item:
                         image_data, content_type = load_url_image_data(image_item["url"], headers) # Pass headers for potential auth

                    if image_data and content_type:
                        upload_metadata = {"model": model, "prompt": form_data.prompt}
                        uploaded_url = upload_image(request, upload_metadata, image_data, content_type, user)
                        if uploaded_url:
                             images.append({"url": uploaded_url})
                        else: log.warning("Failed to upload image data.")
                    else: log.warning("Could not load image data from OpenAI response item.")
            else:
                 log.error("Unexpected response structure from OpenAI API.")
                 raise HTTPException(status_code=500, detail="Unexpected response from OpenAI API.")

            if not images:
                 raise HTTPException(status_code=500, detail="No images were successfully generated or processed.")

            return images

        # *** START OF REPLACED GEMINI BLOCK ***
        elif request.app.state.config.IMAGE_GENERATION_ENGINE == "gemini":
            api_key = request.app.state.config.IMAGES_GEMINI_API_KEY
            # Use form_data.model if provided, else config model, else default
            model = form_data.model or get_image_model(request)
            base_url = request.app.state.config.IMAGES_GEMINI_API_BASE_URL

            if not all([api_key, model, base_url]):
                missing = [name for name, val in [("API Key", api_key), ("Model Name", model), ("Base URL", base_url)] if not val]
                raise HTTPException(status_code=400, detail=f"Gemini configuration missing: {', '.join(missing)}")

            headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

            # Structure the prompt as per Gemini API requirements (may vary by model)
            # Assuming a simple text prompt suffices for generateContent
            structured_prompt = f"{form_data.prompt}" # Simplified from your version, adjust if specific prefix is needed

            # Prepare payload for generateContent API
            # This structure is common for newer Gemini vision models
            data = {
                "contents": [{"role": "user", "parts": [{"text": structured_prompt}]}],
                 "generationConfig": {
                    "candidateCount": form_data.n if form_data.n > 0 else 1,
                    # Output MimeType might be needed for specific models like Imagen
                    # "outputMimeType": "image/png"
                 }
            }

            # Check if specific model requires 'responseModalities' (like flash experimental)
            # This is a placeholder, adjust the model name check as needed
            if "gemini-1.5-flash" in model: # Example check
                log.info("Adding responseModalities for Flash model.")
                data["generationConfig"]["responseModalities"] = ["IMAGE"] # Request only IMAGE modality

            api_url = f"{base_url.strip('/')}/v1beta/models/{model}:generateContent" # Using v1beta commonly

            log.info(f"Calling Gemini generateContent API: {api_url}")
            log.debug(f"Request Data: {json.dumps(data)}")

            response_text_content = ""
            images = []
            processed_image = False
            r = None # Initialize response object

            try:
                r = await asyncio.to_thread(requests.post, url=api_url, json=data, headers=headers, timeout=180)
                log.debug(f"Response Status Code: {r.status_code}")
                # Avoid logging large response bodies directly unless necessary
                # log.debug(f"Response Text (start): {r.text[:1000]}...")

                r.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
                res = r.json()

                # Process successful response
                if "candidates" in res:
                    for candidate in res.get("candidates", []):
                        if "content" in candidate and "parts" in candidate["content"]:
                            for part in candidate["content"]["parts"]:
                                # Check for inline image data
                                if "inlineData" in part:
                                    inline_data = part["inlineData"]
                                    mime_type = inline_data.get("mimeType", "")
                                    b64_data = inline_data.get("data")

                                    if mime_type.startswith("image/") and b64_data:
                                        log.info(f"Found inline image data (mime: {mime_type})")
                                        try:
                                            image_data, content_type = load_b64_image_data(b64_data)
                                            if image_data and content_type:
                                                # Include model and prompt in metadata
                                                upload_metadata = {"engine": "gemini", "model": model, "prompt": form_data.prompt}
                                                url = upload_image(request, upload_metadata, image_data, content_type, user)
                                                if url:
                                                     images.append({"url": url})
                                                     processed_image = True
                                                     log.info(f"Successfully processed and uploaded image: {url}")
                                                else:
                                                     log.warning("upload_image failed to return a URL.")
                                            else:
                                                 log.warning("load_b64_image_data failed to return valid data/type.")
                                        except Exception as upload_err:
                                            log.error(f"Error processing/uploading Gemini image: {upload_err}", exc_info=True)
                                    else:
                                        log.debug("Skipping inlineData part, not image or missing data.")
                                # Capture text parts if no image has been processed yet (e.g., for error messages)
                                elif "text" in part and not processed_image:
                                    response_text_content += part["text"] + "\n"
                                    log.debug(f"Captured text part: {part['text'][:100]}...")
                # Handle cases where the API call succeeded (2xx) but the response indicates an error
                elif "error" in res:
                     error_info = res["error"]
                     log.error(f"Gemini API returned success status but error structure in body: {error_info}")
                     response_text_content = error_info.get("message", "Unknown API error found in response body")
                else:
                     log.warning("Gemini response missing 'candidates' or 'error' structure.")
                     response_text_content = "Unexpected response structure from Gemini API."


            except requests.exceptions.HTTPError as http_err:
                # Handle 4xx/5xx errors specifically
                log.error(f"HTTP error calling Gemini API: {http_err}", exc_info=True)
                error_detail = f"Gemini API returned status {http_err.response.status_code}"
                try:
                    error_json = http_err.response.json()
                    if "error" in error_json and "message" in error_json["error"]:
                        error_detail = f"Gemini API Error: {error_json['error']['message']}"
                    else:
                        error_detail += f": {http_err.response.text[:200]}" # Include snippet of response
                except json.JSONDecodeError:
                    error_detail += f": {http_err.response.text[:200]}"
                raise HTTPException(status_code=http_err.response.status_code, detail=error_detail)

            except requests.exceptions.RequestException as req_err:
                # Handle other network errors (timeout, connection error, etc.)
                log.error(f"Request to Gemini API failed: {req_err}", exc_info=True)
                error_detail = f"Network error communicating with Gemini API: {req_err}"
                raise HTTPException(status_code=500, detail=error_detail)

            # After trying, check if any image was successfully processed
            if not processed_image:
                # If we captured text, use it as the error detail, otherwise use a generic message
                final_error_detail = response_text_content.strip() if response_text_content else "No image data found in the response from Gemini API."
                log.error(f"Gemini image generation failed. Detail: '{final_error_detail}'.")
                # Determine appropriate status code (e.g., 400 if API indicated failure via text, 500 if unexpected structure)
                status_code = 400 if response_text_content else 500
                raise HTTPException(status_code=status_code, detail=final_error_detail)

            # Return the list of successfully processed image URLs
            return images
        # *** END OF REPLACED GEMINI BLOCK ***

        elif request.app.state.config.IMAGE_GENERATION_ENGINE == "comfyui":
            # Determine model, using default if necessary
            model = form_data.model or request.app.state.config.IMAGE_GENERATION_MODEL or "" # Use config default if any

            # Prepare ComfyUI specific data
            data = {
                "prompt": form_data.prompt,
                "width": width,
                "height": height,
                "n": form_data.n if form_data.n > 0 else 1,
            }

            if request.app.state.config.IMAGE_STEPS is not None:
                data["steps"] = request.app.state.config.IMAGE_STEPS

            if form_data.negative_prompt is not None:
                data["negative_prompt"] = form_data.negative_prompt

            # Validate workflow configuration
            if not request.app.state.config.COMFYUI_WORKFLOW or not request.app.state.config.COMFYUI_WORKFLOW_NODES:
                 raise HTTPException(status_code=500, detail="ComfyUI workflow or nodes not configured.")

            # Prepare the form data for comfyui_generate_image
            comfy_form_data = ComfyUIGenerateImageForm(
                workflow=ComfyUIWorkflow(
                    workflow=request.app.state.config.COMFYUI_WORKFLOW,
                    nodes=request.app.state.config.COMFYUI_WORKFLOW_NODES,
                ),
                **data, # Include prompt, size, n, steps, negative_prompt
            )

            res = await comfyui_generate_image(
                model, # Pass the determined model
                comfy_form_data,
                user.id,
                request.app.state.config.COMFYUI_BASE_URL,
                request.app.state.config.COMFYUI_API_KEY,
            )
            log.debug(f"ComfyUI Response: {res}")

            images = []
            if "data" in res and isinstance(res["data"], list):
                headers = None
                if request.app.state.config.COMFYUI_API_KEY:
                    headers = {"Authorization": f"Bearer {request.app.state.config.COMFYUI_API_KEY}"}

                for image_item in res["data"]:
                     if "url" in image_item:
                        image_data, content_type = load_url_image_data(image_item["url"], headers)
                        if image_data and content_type:
                            upload_metadata = comfy_form_data.model_dump(exclude_none=True)
                            upload_metadata["engine"] = "comfyui"
                            upload_metadata["model"] = model # Add model used
                            uploaded_url = upload_image(request, upload_metadata, image_data, content_type, user)
                            if uploaded_url:
                                 images.append({"url": uploaded_url})
                            else: log.warning("Failed to upload ComfyUI image.")
                        else: log.warning("Could not load image data from ComfyUI URL.")
                     else: log.warning("ComfyUI response item missing 'url'.")
            else:
                 log.error("Unexpected response structure from ComfyUI.")
                 # Consider raising or returning error based on 'res' content if available
                 error_detail = res.get("error", "Unexpected response from ComfyUI.") if isinstance(res, dict) else "Unexpected response from ComfyUI."
                 raise HTTPException(status_code=500, detail=error_detail)


            if not images:
                 raise HTTPException(status_code=500, detail="No ComfyUI images were successfully generated or processed.")

            return images

        elif (
            request.app.state.config.IMAGE_GENERATION_ENGINE == "automatic1111"
            or request.app.state.config.IMAGE_GENERATION_ENGINE == ""
        ):
            # Set model if provided in form_data, otherwise it uses the currently set one
            if form_data.model:
                set_image_model(request, form_data.model)
                current_model = form_data.model
            else:
                # Get current model to store in metadata, handle potential fetch failure
                current_model = get_image_model(request)
                if current_model is None:
                     raise HTTPException(status_code=500, detail="Failed to determine current Automatic1111 model.")


            data = {
                "prompt": form_data.prompt,
                "batch_size": form_data.n if form_data.n > 0 else 1,
                "width": width,
                "height": height,
            }

            # Add optional parameters from config if they exist
            if request.app.state.config.IMAGE_STEPS is not None:
                data["steps"] = request.app.state.config.IMAGE_STEPS
            if form_data.negative_prompt is not None:
                data["negative_prompt"] = form_data.negative_prompt
            if request.app.state.config.AUTOMATIC1111_CFG_SCALE:
                data["cfg_scale"] = request.app.state.config.AUTOMATIC1111_CFG_SCALE
            if request.app.state.config.AUTOMATIC1111_SAMPLER:
                data["sampler_name"] = request.app.state.config.AUTOMATIC1111_SAMPLER
            if request.app.state.config.AUTOMATIC1111_SCHEDULER:
                data["scheduler"] = request.app.state.config.AUTOMATIC1111_SCHEDULER


            api_url = f"{request.app.state.config.AUTOMATIC1111_BASE_URL.strip('/')}/sdapi/v1/txt2img"
            auth_header = get_automatic1111_api_auth(request)

            r = await asyncio.to_thread(requests.post, url=api_url, json=data, headers={"authorization": auth_header}, timeout=180) # Add timeout
            r.raise_for_status()
            res = r.json()
            log.debug(f"Automatic1111 Response: {res}")

            images = []
            if "images" in res and isinstance(res["images"], list):
                info_str = res.get("info", "{}") # Get info string, default to empty JSON
                try:
                     # Parse info string once for metadata
                     info_data = json.loads(info_str)
                except json.JSONDecodeError:
                     log.warning("Could not parse Automatic1111 info string.")
                     info_data = {}


                for b64_image_str in res["images"]:
                    image_data, content_type = load_b64_image_data(b64_image_str)
                    if image_data and content_type:
                         # Combine request data and parsed info for metadata
                        upload_metadata = {**data, "info": info_data, "engine": "automatic1111", "model": current_model}
                        # Remove sensitive or large items if necessary e.g., upload_metadata.pop('prompt', None)
                        uploaded_url = upload_image(request, upload_metadata, image_data, content_type, user)
                        if uploaded_url:
                            images.append({"url": uploaded_url})
                        else: log.warning("Failed to upload Automatic1111 image.")
                    else: log.warning("Could not load base64 image data from Automatic1111 response.")
            else:
                 log.error("Unexpected response structure from Automatic1111 API.")
                 raise HTTPException(status_code=500, detail="Unexpected response from Automatic1111 API.")


            if not images:
                 raise HTTPException(status_code=500, detail="No Automatic1111 images were successfully generated or processed.")

            return images
        else:
             # Handle case where engine is enabled but not recognized
             raise HTTPException(status_code=400, detail=f"Unsupported image generation engine: {request.app.state.config.IMAGE_GENERATION_ENGINE}")

    except requests.exceptions.RequestException as req_err:
         # Catch network errors specifically
         log.error(f"Network error during image generation request: {req_err}", exc_info=True)
         error_detail = f"Network error: {req_err}"
         status_code = 503 # Service Unavailable or Bad Gateway might be appropriate
         if req_err.response is not None:
              status_code = req_err.response.status_code
              try:
                   # Try to get more specific error from response body
                   err_json = req_err.response.json()
                   if "error" in err_json:
                        if isinstance(err_json["error"], dict) and "message" in err_json["error"]:
                             error_detail = err_json["error"]["message"]
                        else: error_detail = str(err_json["error"])
                   elif "detail" in err_json: # Some APIs use 'detail'
                        error_detail = err_json["detail"]
                   else: error_detail = req_err.response.text[:200] # Fallback to text
              except json.JSONDecodeError:
                   error_detail = req_err.response.text[:200]

         raise HTTPException(status_code=status_code, detail=ERROR_MESSAGES.DEFAULT(error_detail))

    except HTTPException as http_exc:
         # Re-raise HTTPExceptions that were raised intentionally within the blocks
         raise http_exc
    except Exception as e:
        # Catch any other unexpected errors
        log.exception(f"Unexpected error during image generation: {e}") # Log full traceback
        error_detail = f"An unexpected error occurred: {e}"
        # Try to get error details from response if available
        if r is not None and hasattr(r, 'status_code') and r.status_code >= 400:
            try:
                data = r.json()
                if "error" in data:
                     if isinstance(data["error"], dict) and "message" in data["error"]:
                          error_detail = data["error"]["message"]
                     else: error_detail = str(data["error"])
                elif "detail" in data:
                     error_detail = data["detail"]
                else: error_detail = r.text[:200]

            except json.JSONDecodeError:
                 error_detail = r.text[:200] if hasattr(r, 'text') else f"An unexpected error occurred: {e}"


        raise HTTPException(status_code=500, detail=ERROR_MESSAGES.DEFAULT(error_detail))