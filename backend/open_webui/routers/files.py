import logging
import os
import uuid
from fnmatch import fnmatch # Keeping this as it was in the original for search
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
    Query, # Keeping this as it was in the original for search/list
)
from fastapi.responses import FileResponse, StreamingResponse
from open_webui.constants import ERROR_MESSAGES
from open_webui.env import SRC_LOG_LEVELS
from open_webui.models.files import (
    FileForm,
    FileModel, # Keeping this as it was in the original
    FileModelResponse,
    Files,
)
from open_webui.models.knowledge import Knowledges # Keeping this as it was in the original

# Keeping these as they were in the original
from open_webui.routers.knowledge import get_knowledge, get_knowledge_list
from open_webui.routers.retrieval import ProcessFileForm, process_file
from open_webui.routers.audio import transcribe
from open_webui.storage.provider import Storage
from open_webui.utils.auth import get_admin_user, get_verified_user
from pydantic import BaseModel

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])


router = APIRouter()


############################
# Check if the current user has access to a file through any knowledge bases the user may be in.
# Keeping this function as it was in the original file you provided
############################


def has_access_to_file(
    file_id: Optional[str], access_type: str, user=Depends(get_verified_user)
) -> bool:
    file = Files.get_file_by_id(file_id)
    log.debug(f"Checking if user has {access_type} access to file")

    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    has_access = False
    knowledge_base_id = file.meta.get("collection_name") if file.meta else None

    if knowledge_base_id:
        knowledge_bases = Knowledges.get_knowledge_bases_by_user_id(
            user.id, access_type
        )
        for knowledge_base in knowledge_bases:
            if knowledge_base.id == knowledge_base_id:
                has_access = True
                break

    return has_access


############################
# Upload File
############################


@router.post("/", response_model=FileModelResponse)
def upload_file(
    request: Request,
    file: UploadFile = File(...),
    user=Depends(get_verified_user),
    file_metadata: dict = {},
    process: bool = Query(True), # Keeping this parameter from original
):
    # Using your log line here as it's slightly different
    log.info(f"upload_file called. file.content_type: {file.content_type}")
    try:
        unsanitized_filename = file.filename
        filename = os.path.basename(unsanitized_filename)

        id = str(uuid.uuid4())
        name = filename
        filename = f"{id}_{filename}"
        contents, file_path = Storage.upload_file(file.file, filename)

        # Define file_meta using original structure's fields
        file_meta = {
            "name": name,
            "content_type": file.content_type,
            "size": len(contents),
            "data": file_metadata,
        }

        file_item = Files.insert_new_file(
            user.id,
            FileForm(
                **{
                    "id": id,
                    "filename": name,
                    "path": file_path,
                    "meta": file_meta,
                }
            ),
        )

        # *** Start of Your Modified Processing Block ***
        # Note: We still check the original 'process' flag first
        if process:
            try:
                current_content_type = file.content_type
                if current_content_type and current_content_type.startswith("audio/"):
                    log.info(f"Processing audio file: {id}")
                    # Use the stored file_path directly, Storage.get_file might be redundant if upload returns usable path
                    stored_file_path = Storage.get_file(file_path) # Keep get_file as per original logic flow before your change
                    result = transcribe(request, stored_file_path)
                    log.info(f"Transcription result for {id}: {result.get('text', '')[:100]}...") # Log snippet
                    process_file(
                        request,
                        ProcessFileForm(file_id=id, content=result.get("text", "")),
                        user=user,
                    )
                # Skip Images and Videos for RAG processing (Your Change)
                elif current_content_type and current_content_type.startswith(("image/", "video/")):
                     log.info(f"Skipping RAG processing for non-document file type ({current_content_type}): {id}")
                     pass # Explicitly do nothing for image/video RAG
                else:
                    # Process other files (potential documents) for RAG
                    log.info(f"Processing potential document file for RAG (type: {current_content_type}): {id}")
                    process_file(request, ProcessFileForm(file_id=id), user=user)

                # Refresh file_item after processing
                file_item = Files.get_file_by_id(id=id)

            except Exception as e:
                # Your error handling for processing failure
                log.error(f"Error during post-upload processing for file: {id}", exc_info=True)
                # Ensure file_item exists before trying to access it for dump
                if not file_item:
                    file_item = Files.get_file_by_id(id=id) # Try refetching if initial insert succeeded but processing failed early

                file_item_dump = file_item.model_dump() if file_item else {}
                # Return the file item model response with error attribute set
                # This structure is slightly different from original's error handling here
                return FileModelResponse(
                    **file_item_dump,
                    error=f"File uploaded but processing failed: {str(e.detail) if hasattr(e, 'detail') else str(e)}",
                )
        # *** End of Your Modified Processing Block ***


        # Your final return logic which handles potential error attribute
        if file_item:
            # Check if the error attribute was set during processing
            if hasattr(file_item, 'error') and file_item.error:
                 log.warning(f"Returning file item with processing error attached: {file_item.error}")
            else:
                 log.info(f"File upload and processing completed successfully for ID: {id}")
            # Ensure we return the correct type (FileModelResponse)
            # If file_item is already FileModelResponse (from error handling), use it, otherwise create one
            if isinstance(file_item, FileModelResponse):
                 return file_item
            else:
                 return FileModelResponse(**file_item.model_dump())

        else:
            # Your error handling if file_item is None after insert/processing attempt
            log.error(f"Failed to retrieve file item after upload for ID: {id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, # Changed to 500 as per your version
                detail=ERROR_MESSAGES.DEFAULT("Error retrieving file after upload"),
            )

    except Exception as e:
        # Your more generic outer error handling
        log.error("Error during file upload process", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(f"Error uploading file: {e}"),
        )


############################
# List Files
# Keeping original logic with 'content' parameter
############################


@router.get("/", response_model=list[FileModelResponse])
async def list_files(user=Depends(get_verified_user), content: bool = Query(True)):
    if user.role == "admin":
        files = Files.get_files()
    else:
        files = Files.get_files_by_user_id(user.id)

    if not content:
        for file in files:
            if hasattr(file, 'data') and isinstance(file.data, dict) and 'content' in file.data:
                 del file.data["content"] # Safely delete content if present

    # Convert FileModel to FileModelResponse if necessary (assuming Files returns FileModel)
    response_files = [FileModelResponse(**f.model_dump()) for f in files]
    return response_files


############################
# Search Files
# Keeping original function
############################


@router.get("/search", response_model=list[FileModelResponse])
async def search_files(
    filename: str = Query(
        ...,
        description="Filename pattern to search for. Supports wildcards such as '*.txt'",
    ),
    content: bool = Query(True),
    user=Depends(get_verified_user),
):
    """
    Search for files by filename with support for wildcard patterns.
    """
    # Get files according to user role
    if user.role == "admin":
        files = Files.get_files()
    else:
        files = Files.get_files_by_user_id(user.id)

    # Get matching files
    matching_files = [
        file for file in files if fnmatch(file.filename.lower(), filename.lower())
    ]

    if not matching_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No files found matching the pattern.",
        )

    if not content:
        for file in matching_files:
             if hasattr(file, 'data') and isinstance(file.data, dict) and 'content' in file.data:
                del file.data["content"] # Safely delete content if present

    # Convert FileModel to FileModelResponse if necessary
    response_files = [FileModelResponse(**f.model_dump()) for f in matching_files]
    return response_files


############################
# Delete All Files
# Keeping original function
############################


@router.delete("/all")
async def delete_all_files(user=Depends(get_admin_user)):
    result = Files.delete_all_files()
    if result:
        try:
            Storage.delete_all_files()
        except Exception as e:
            log.exception(e)
            log.error("Error deleting files")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT("Error deleting files"),
            )
        return {"message": "All files deleted successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("Error deleting files"),
        )


############################
# Get File By Id
# Keeping original function with has_access_to_file check
############################


@router.get("/{id}", response_model=Optional[FileModel]) # Original uses FileModel here
async def get_file_by_id(id: str, user=Depends(get_verified_user)):
    file = Files.get_file_by_id(id)

    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        file.user_id == user.id
        or user.role == "admin"
        or has_access_to_file(id, "read", user) # Original check
    ):
        return file # Return FileModel as per original response_model
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


############################
# Get File Data Content By Id
# Keeping original function with has_access_to_file check
############################


@router.get("/{id}/data/content")
async def get_file_data_content_by_id(id: str, user=Depends(get_verified_user)):
    file = Files.get_file_by_id(id)

    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        file.user_id == user.id
        or user.role == "admin"
        or has_access_to_file(id, "read", user) # Original check
    ):
        # Ensure data exists and is a dict before accessing 'content'
        content = file.data.get("content", "") if isinstance(file.data, dict) else ""
        return {"content": content}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


############################
# Update File Data Content By Id
# Keeping original function with has_access_to_file check
############################


class ContentForm(BaseModel):
    content: str


@router.post("/{id}/data/content/update")
async def update_file_data_content_by_id(
    request: Request, id: str, form_data: ContentForm, user=Depends(get_verified_user)
):
    file = Files.get_file_by_id(id)

    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        file.user_id == user.id
        or user.role == "admin"
        or has_access_to_file(id, "write", user) # Original check
    ):
        try:
            process_file(
                request,
                ProcessFileForm(file_id=id, content=form_data.content),
                user=user,
            )
            file = Files.get_file_by_id(id=id) # Re-fetch file
        except Exception as e:
            log.exception(e)
            log.error(f"Error processing file: {file.id}")
            # Consider re-raising or returning error status? Original didn't specify error return here.

        # Ensure file and data exist before returning content
        content = ""
        if file and isinstance(file.data, dict):
             content = file.data.get("content", "")

        return {"content": content}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


############################
# Get File Content By Id
# Keeping original function with has_access_to_file check and attachment parameter
############################


@router.get("/{id}/content")
async def get_file_content_by_id(
    id: str, user=Depends(get_verified_user), attachment: bool = Query(False) # Original parameter
):
    file = Files.get_file_by_id(id)

    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        file.user_id == user.id
        or user.role == "admin"
        or has_access_to_file(id, "read", user) # Original check
    ):
        try:
            file_path = Storage.get_file(file.path)
            file_path = Path(file_path)

            if file_path.is_file():
                filename = file.meta.get("name", file.filename)
                encoded_filename = quote(filename)
                content_type = file.meta.get("content_type")
                headers = {}

                # Original logic for Content-Disposition based on attachment flag and type
                if attachment:
                    headers["Content-Disposition"] = (
                        f"attachment; filename*=UTF-8''{encoded_filename}"
                    )
                else:
                    # Inline PDF specifically
                    if content_type == "application/pdf" or filename.lower().endswith(".pdf"):
                        headers["Content-Disposition"] = (
                            f"inline; filename*=UTF-8''{encoded_filename}"
                        )
                        content_type = "application/pdf" # Ensure correct media type for inline PDF
                    # Default to attachment for other non-plain-text types if not forced by attachment=True
                    elif content_type != "text/plain":
                         # This logic might need review based on desired behavior for images/etc when attachment=False
                         # Original seemed to imply attachment for non-PDF/non-text, let's keep that
                         headers["Content-Disposition"] = (
                             f"attachment; filename*=UTF-8''{encoded_filename}"
                         )
                         # If attachment=False, but it's not PDF/text, should media_type be None or original?
                         # Let's pass the original content_type


                return FileResponse(file_path, headers=headers, media_type=content_type)
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=ERROR_MESSAGES.NOT_FOUND + " (File not found in storage)",
                )
        except Exception as e:
            log.exception(e)
            log.error("Error getting file content")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT("Error getting file content"),
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


# Keeping original /content/html endpoint
@router.get("/{id}/content/html")
async def get_html_file_content_by_id(id: str, user=Depends(get_verified_user)):
    file = Files.get_file_by_id(id)

    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        file.user_id == user.id
        or user.role == "admin"
        or has_access_to_file(id, "read", user) # Original check
    ):
        try:
            file_path = Storage.get_file(file.path)
            file_path = Path(file_path)

            if file_path.is_file():
                log.info(f"Serving HTML file_path: {file_path}") # Original log line had different content
                # Assuming HTML should always be served directly if possible
                return FileResponse(file_path, media_type="text/html") # Explicitly set media type
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=ERROR_MESSAGES.NOT_FOUND + " (File not found in storage)",
                )
        except Exception as e:
            log.exception(e)
            log.error("Error getting HTML file content")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT("Error getting HTML file content"),
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


# Removing the duplicate /content/{file_name} endpoint as it wasn't in your 'fixed' version
# and the primary /content endpoint handles naming/disposition.
# If the original truly had *both* /content and /content/{file_name} with different logic,
# we might need to reconsider, but usually one is sufficient.


############################
# Delete File By Id
# Keeping original function with has_access_to_file check
############################


@router.delete("/{id}")
async def delete_file_by_id(id: str, user=Depends(get_verified_user)):
    file = Files.get_file_by_id(id)

    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        file.user_id == user.id
        or user.role == "admin"
        or has_access_to_file(id, "write", user) # Original check
    ):
        # TODO: Add Chroma cleanup here (as per original comment)

        result = Files.delete_file_by_id(id)
        if result:
            try:
                # Make sure file.path exists before deleting
                if file.path:
                    Storage.delete_file(file.path)
                else:
                    log.warning(f"File {id} deleted from DB but had no path in storage.")
            except Exception as e:
                log.exception(e)
                # Log error but maybe don't raise? Deletion from DB succeeded.
                # Original raises, let's keep that for consistency.
                log.error(f"Error deleting file {file.path} from storage for ID {id}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=ERROR_MESSAGES.DEFAULT("Error deleting file from storage"),
                )
            return {"message": "File deleted successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT("Error deleting file from database"),
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )