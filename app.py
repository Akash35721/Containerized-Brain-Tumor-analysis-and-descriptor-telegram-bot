# app_with_followup.py
# Telegram bot that processes brain tumour images with YOLOv8,
# explains them via Gemma3 on Ollama, and handles follow-up questions.

import os
import tempfile
import logging
import shutil
import ollama
import io # Needed for BytesIO

from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# --- Configuration ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7896722959:AAFzIaCvAtURfQNKeM6QQmLDHmGwgxmWiiQ") 
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "C:/Users/2004a/OneDrive - UPES/SEMESTER 6/non sharable/LLM/final_1_yolo_model/runs/detect/train2/weights/best.pt")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:latest")
YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.70"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", None)

# --- Logging Setup ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Basic Validation & Initialization ---
if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN" or not TELEGRAM_BOT_TOKEN.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
    logger.critical("!!!!!!!!!!!!!! Telegram Bot Token seems invalid, is a placeholder, or is not set! Set the TELEGRAM_BOT_TOKEN environment variable.")
    exit() # Exit if no valid token

if not os.path.isfile(YOLO_MODEL_PATH):
    logger.critical(f"!!!!!!! YOLO weights not found at: {YOLO_MODEL_PATH}. Bot cannot start.")
    exit() # Exit if model not found

# Load YOLO Model
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    logger.info(f"<<<<>>>> Loaded YOLO model '{YOLO_MODEL_PATH}' with classes: {yolo_model.names}<<<<>>>> ")
except Exception as e:
    logger.exception(f"!!!!!!!Failed to load YOLO model: {e}!!!!!!!")
    raise # Stop if YOLO model fails to load

# Initialize Ollama Client
ollama_client = None
if OLLAMA_MODEL: # Only try to connect if a model is specified
    try:
        logger.info(f"Attempting to initialize Ollama client for model '{OLLAMA_MODEL}' at host: {OLLAMA_HOST or 'default'}...")
        ollama_client = ollama.Client(host=OLLAMA_HOST)
        ollama_client.list() # Check connection
        logger.info(f"<<<<>>>> Initialized Ollama client. Base URL: {ollama_client._client.base_url} <<<<>>>>")
    except Exception as e:
        logger.warning(f"!!!!!!! Failed to initialize or connect to Ollama client (Host: '{OLLAMA_HOST or 'default'}', Model: '{OLLAMA_MODEL}'). Is Ollama running and the model pulled? Error: {e}. Explanations will be unavailable.!!!!!!!")
        ollama_client = None
else:
     logger.warning("!!!!!!! OLLAMA_MODEL not set. Explanations will be unavailable. !!!!!!!")

# --- Helper Function for Annotation ---

def annotate_image(image_bytes: bytes, boxes_data: list, class_names: dict, font: ImageFont.FreeTypeFont) -> bytes:
    """Annotates an image in memory based on detection data."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)

        if not boxes_data: # Should not happen if called correctly, but safe check
             logger.warning("annotate_image called with no boxes_data.")
             return image_bytes # Return original if no boxes

        for *box, conf, cls in boxes_data:
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            class_name = class_names.get(class_id, f"Unknown Class {class_id}")
            label = f"{class_name} {conf:.2f}"

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            try:
                # Get text size using getbbox
                text_bbox = font.getbbox(label)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                # Simple text placement; adjust as needed
                text_y = y1 - text_height - 2 if y1 > (text_height + 5) else y2 + 2
                text_x = x1

                # Optional: Add background rectangle for better readability
                # draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], fill="red")
                draw.text((text_x, text_y), label, font=font, fill="red")
            except Exception as e:
                 logger.error(f"Error drawing text/bbox for label '{label}': {e}")


        # Save annotated image to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        annotated_bytes = img_byte_arr.getvalue()
        return annotated_bytes
    except Exception as e:
        logger.exception(f"Error during image annotation in memory: {e}")
        return image_bytes # Return original on error

# --- Telegram Bot Handlers ---

async def start(update: Update, context: CallbackContext) -> None:
    """Sends a welcome message."""
    await update.message.reply_text(
        ":>) Hello! Send me a brain tumour image (like an MRI scan). I'll detect potential tumours, mark them, and provide an explanation using an AI model via Ollama. You can ask one follow-up question after the analysis."
    )
    # Clear any leftover context from previous interactions in this chat
    context.chat_data.pop('last_analysis_context', None)


async def handle_image(update: Update, context: CallbackContext) -> None:
    """Handles incoming photos, runs YOLO, annotates, sends to Ollama, and stores context."""
    if not update.message.photo:
        logger.warning("handle_image called without photo.")
        return

    # Clear context from previous analysis in this chat before starting new one
    context.chat_data.pop('last_analysis_context', None)
    logger.debug(f"Cleared previous context for chat_id: {update.effective_chat.id}")

    photo = update.message.photo[-1] # Highest resolution
    photo_file_id = photo.file_id # Store file_id for potential follow-up

    # Use a temporary directory for processing this image
    tmpdir = tempfile.mkdtemp()
    logger.info(f"Created temp directory: {tmpdir} for chat_id: {update.effective_chat.id}")
    img_path = os.path.join(tmpdir, "input.jpg")
    # labelled_path = os.path.join(tmpdir, "labelled.jpg") # We will annotate in memory now

    try:
        # 1. Download Image
        logger.info(f"Downloading image file_id: {photo_file_id}")
        file = await photo.get_file()
        await file.download_to_drive(img_path)
        logger.info(f"Image saved to: {img_path}")

        # Read image bytes for in-memory annotation later
        with open(img_path, "rb") as f:
            original_image_bytes = f.read()

        # 2. Run YOLO Inference
        logger.info(f"Running YOLO detection (model: {YOLO_MODEL_PATH}, conf: {YOLO_CONF_THRESHOLD})")
        results = yolo_model.predict(
            source=img_path,
            save=False,
            conf=YOLO_CONF_THRESHOLD
        )
        result = results[0]
        raw_boxes = result.boxes.data.tolist() # [x1, y1, x2, y2, conf, cls]
        logger.info(f"Raw YOLO detections: {raw_boxes}")

        # 3. Prepare for Annotation
        try:
            font = ImageFont.load_default(size=15)
        except OSError:
            logger.warning("Default font not found, using basic fallback.")
            font = ImageFont.load_default()

        # 4. Handle Detections / No Detections
        if not raw_boxes:
            await update.message.reply_text(
                f"<<<<>>>> Scan processed. No potential tumours detected above the confidence threshold ({YOLO_CONF_THRESHOLD}).<<<<>>>>"
            )
            # No context to save, just return
            return

        # --- Detections Found ---
        detections_found = True
        detected_classes_summary = []
        for *box, conf, cls in raw_boxes:
            class_id = int(cls)
            class_name = yolo_model.names.get(class_id, f"Unknown Class {class_id}")
            detected_classes_summary.append(f"{class_name} (conf: {conf:.2f})")

        # 5. Annotate Image (In Memory)
        logger.info("Annotating image in memory...")
        annotated_image_bytes = annotate_image(
            original_image_bytes,
            raw_boxes,
            yolo_model.names,
            font
        )

        # 6. Send Annotated Image back to Telegram
        logger.info("Sending annotated photo back to user.")
        caption_text = f"( :; ) Potential tumour detections: {', '.join(detected_classes_summary)}. (Threshold ≥ {YOLO_CONF_THRESHOLD})"
        await update.message.reply_photo(
            photo=InputFile(io.BytesIO(annotated_image_bytes), filename="labelled_scan.jpg"),
            caption=caption_text
        )

        # 7. Call Ollama for Explanation (only if detections were found and client is available)
        explanation = None # Initialize explanation
        if detections_found and ollama_client:
            status_message = await update.message.reply_text("??? Analyzing detected areas using AI...")
            logger.info(f"Sending image and prompt to Ollama model '{OLLAMA_MODEL}'.")

   
            

           
            prompt_for_ollama = f"""You are an analytical AI assistant interpreting findings on a provided brain MRI scan image.
This image has been pre-processed by an object detection model, and potential areas of interest are marked with red boxes and associated labels (e.g., 'classname confidence_score').

The preliminary detections reported were: {', '.join(detected_classes_summary) or 'None'}.

Your task is to analyze the visual content within these marked regions and provide informative insights based **only** on the visual evidence in the image. Please:

1.  **Describe the visual characteristics** of the areas marked by the red boxes. Consider aspects like:
    * Appearance (e.g., bright, dark, textured, uniform) relative to surrounding tissue.
    * Shape and border definition (e.g., round, irregular, well-defined, infiltrative).
    * Approximate location if discernible anatomical landmarks are visible near the box.
    * Any other notable visual features within the box (e.g., signs of swelling, mass effect if visible).

2.  **Provide general context** about what findings with these visual characteristics and labels ('{', '.join(detected_classes_summary) or 'None'}') might typically represent in the context of brain MRI analysis. Keep this general and informative.

Do not attempt to definitively diagnose. Focus solely on describing the visual information presented in the marked areas and providing relevant general context.

**IMPORTANT:** This is an AI analysis for informational purposes only and cannot replace professional medical evaluation or diagnosis."""

            try:
                # Send the ANNOTATED image bytes to Ollama
                # We need to save the annotated bytes to a temporary file path for Ollama client
                # Alternatively, if ollama client supports bytes directly, use that.
                # Current ollama library requires a path.
                temp_annotated_path = os.path.join(tmpdir, "ollama_temp_annotated.jpg")
                with open(temp_annotated_path, "wb") as f_annotated:
                    f_annotated.write(annotated_image_bytes)

                response = ollama_client.chat(
                    model=OLLAMA_MODEL,
                    messages=[
                        {
                            'role': 'user',
                            'content': prompt_for_ollama,
                            'images': [temp_annotated_path] # Pass the path to the temp annotated file
                        }
                    ],
                    options={'temperature': 0.3}    #<<==========================================================TEMPERATURE
                )
                explanation = response['message']['content']
                logger.info("Ollama analysis successful.")
                await status_message.edit_text(explanation + "\n\n*You can ask one follow-up question about this analysis.*")

            except Exception as e:
                logger.exception(f"!!!!!!! Error contacting Ollama API: {e}!!!!!!!")
                explanation = (
                    "Sorry, I couldn't get an explanation from the AI model. "
                    f"(Error type: {type(e).__name__})"
                )
                await status_message.edit_text(explanation)

            # 8. Store Context for Follow-up (if explanation was successful)
            if explanation and "Sorry, I couldn't get an explanation" not in explanation:
                context.chat_data['last_analysis_context'] = {
                    'original_photo_file_id': photo_file_id,
                    'raw_boxes': raw_boxes, # Store the detection data
                    'class_names': yolo_model.names, # Store class names map
                    'initial_explanation': explanation,
                    'detected_classes_summary': detected_classes_summary, # Store for context
                    'timestamp': update.message.date # Store time for potential expiry
                }
                logger.info(f"Stored analysis context for chat_id: {update.effective_chat.id}")

        elif detections_found and not ollama_client:
            await update.message.reply_text("!!!!!!! Detections found, but AI explanation service is unavailable.")
        # No explicit message needed if no detections were found (handled earlier)

    except Exception as e:
        logger.exception(f"An error occurred processing the image in chat {update.effective_chat.id}: {e}")
        await update.message.reply_text("!!!!!!! Apologies, an unexpected error occurred while processing your image.")
        context.chat_data.pop('last_analysis_context', None) # Clear context on error

    finally:
        # Clean up temporary directory
        try:
            if os.path.exists(tmpdir):
                shutil.rmtree(tmpdir)
                logger.info(f"Cleaned up temp directory: {tmpdir}")
        except Exception as e:
            logger.error(f"Error cleaning up temp directory '{tmpdir}': {e}")


async def handle_generic_message(update: Update, context: CallbackContext) -> None:
    """Handles non-command text messages, potentially as follow-ups."""
    user_message = update.message.text
    chat_id = update.effective_chat.id
    last_context = context.chat_data.get('last_analysis_context')

    # Check if there's context AND Ollama is available
    if last_context and ollama_client:
        logger.info(f"Handling potential follow-up message in chat_id: {chat_id}")

        # Optional: Check timestamp if you want context to expire
        # from datetime import datetime, timedelta
        # if datetime.now() - last_context['timestamp'] > timedelta(minutes=5):
        #     logger.info(f"Analysis context expired for chat_id: {chat_id}")
        #     await update.message.reply_text("The previous analysis context has expired. Please send a new image.")
        #     context.chat_data.pop('last_analysis_context', None)
        #     return

        # --- Handle the Follow-up ---
        original_photo_file_id = last_context['original_photo_file_id']
        raw_boxes = last_context['raw_boxes']
        class_names = last_context['class_names']
        initial_explanation = last_context['initial_explanation']
        detected_classes_summary = last_context['detected_classes_summary']

        # Clear the context immediately after retrieving it, allowing only one follow-up
        context.chat_data.pop('last_analysis_context')
        logger.info(f"Cleared analysis context after retrieving for follow-up in chat_id: {chat_id}")

        status_message = await update.message.reply_text("??? Processing your follow-up question with AI...")

        # Need to re-acquire the image and re-annotate it for the multimodal model
        tmpdir_followup = tempfile.mkdtemp()
        followup_img_path = os.path.join(tmpdir_followup, "followup_original.jpg")
        followup_annotated_path = os.path.join(tmpdir_followup, "followup_annotated.jpg")

        try:
            # 1. Re-download the original image
            logger.info(f"Re-downloading original image file_id: {original_photo_file_id} for follow-up.")
            file = await context.bot.get_file(original_photo_file_id)
            await file.download_to_drive(followup_img_path)

            # 2. Re-annotate the image (using stored boxes)
            logger.info("Re-annotating image in memory for follow-up...")
            with open(followup_img_path, "rb") as f:
                original_image_bytes = f.read()

            try:
                 # Ensure font is loaded again if needed (or pass it)
                 font = ImageFont.load_default(size=15)
            except OSError:
                 font = ImageFont.load_default()

            annotated_bytes_followup = annotate_image(
                original_image_bytes,
                raw_boxes,
                class_names,
                font
            )
            with open(followup_annotated_path, "wb") as f_annotated:
                 f_annotated.write(annotated_bytes_followup)
            logger.info(f"Re-annotated image saved to: {followup_annotated_path}")


            # 3. Construct Follow-up Prompt
            followup_prompt = f"""You are an analytical assistant. Previously, you analyzed an image (which I am providing again with annotations) with initial findings: '{', '.join(detected_classes_summary) or 'None'}'.
Your initial explanation was:
"{initial_explanation}"

Now, the user has a follow-up question: "{user_message}"

Based ONLY on the provided image, the initial analysis context, and the user's follow-up question, please provide a concise and relevant answer to the follow-up. Maintain the same professional tone and adhere to the original instructions regarding medical context and disclaimers if applicable."""

            # 4. Call Ollama
            logger.info(f"Sending follow-up prompt and image to Ollama model '{OLLAMA_MODEL}'.")
            response = ollama_client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {
                        'role': 'user',
                        'content': followup_prompt,
                        'images': [followup_annotated_path] # Send the re-annotated image path
                    }
                ],
                 options={'temperature': 0.3}
            )
            followup_explanation = response['message']['content']
            logger.info("Ollama follow-up analysis successful.")
            await status_message.edit_text(followup_explanation)

        except Exception as e:
            logger.exception(f"!!!!!!! Error processing follow-up question in chat {chat_id}: {e}!!!!!!!")
            await status_message.edit_text(
                "Sorry, I encountered an error trying to process your follow-up question. "
                f"(Error type: {type(e).__name__})"
            )
        finally:
            # Clean up temporary directory for the follow-up
            try:
                if os.path.exists(tmpdir_followup):
                    shutil.rmtree(tmpdir_followup)
                    logger.info(f"Cleaned up follow-up temp directory: {tmpdir_followup}")
            except Exception as e:
                logger.error(f"Error cleaning up follow-up temp directory '{tmpdir_followup}': {e}")

    # If no context or Ollama is unavailable, treat as regular text
    else:
        logger.info(f"Handling generic text message (no context or Ollama unavailable) in chat_id: {chat_id}")
        await update.message.reply_text(
            "I can only process images or a single follow-up question immediately after an image analysis (if AI explanations are enabled). Please send an image of a brain scan."
        )

# --- Main Function ---
def main() -> None:
    """Starts the bot."""
    # Re-check critical components before starting
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN":
         logger.critical("TOKEN MISSING - exiting.")
         return
    if not yolo_model:
         logger.critical("YOLO MODEL MISSING - exiting.")
         return
    # Warning if Ollama not available, but bot can still run without explanations/follow-ups
    if ollama_client is None:
        logger.warning("!!!!!!!!!!!!!! Ollama client is not available. Bot starting without explanation/follow-up functionality.")

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    # Use handle_generic_message for text now
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_generic_message))

    logger.info("Starting the brain tumor detection bot polling...")
    application.run_polling()

if __name__ == "__main__":
    main()