import os
import cv2
import base64
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import requests
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, Future
from ultralytics import YOLO


# CONFIGURATION

@dataclass
class Config:
    MODEL_PATH: str = "yolov8n.pt"
    UPLOAD_FOLDER: str = "uploads"

    # YOLO settings
    CONFIDENCE_THRESHOLD: float = 0.50

    # Object tracking
    TRACKER_CONFIG: str = "bytetrack.yaml"

    # Product analysis settings
    MAX_WORKERS: int = 3
    API_TIMEOUT: int = 30
    MAX_API_REQUESTS_PER_TRACK: int = 1

    # Do not immediately send every new detection to the API.
    # The object must appear in at least this many frames.
    MIN_TRACK_AGE: int = 5

    # Minimum crop size
    MIN_CROP_SIZE: int = 40


config = Config()


# LOGGING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# PRODUCT EXTRACTOR

class ProductExtractor:

    def __init__(self):

        load_dotenv()

        self.openai_key = os.getenv("OPENAI_KEY")

        if not self.openai_key:
            raise ValueError(
                "OPENAI_KEY is missing. "
                "Add it to your .env file."
            )

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_key}"
        }

        # TO Prevent duplicate API requests for identical images
        self.image_cache: Dict[str, str] = {}


    # TO Encode OpenCV image

    @staticmethod
    def encode_image(image) -> str:

        success, buffer = cv2.imencode(
            ".jpg",
            image
        )

        if not success:
            raise ValueError(
                "Failed to encode image."
            )

        return base64.b64encode(
            buffer
        ).decode("utf-8")


    # To Generate image hash

    @staticmethod
    def get_image_hash(image) -> Optional[str]:

        success, buffer = cv2.imencode(
            ".jpg",
            image
        )

        if not success:
            return None

        return hashlib.md5(
            buffer.tobytes()
        ).hexdigest()


    # To Extract product information

    def extract_product(
        self,
        image,
        yolo_class_name: str
    ) -> str:

        image_hash = self.get_image_hash(image)

        # To Check cache first
        if (
            image_hash
            and image_hash in self.image_cache
        ):
            return self.image_cache[image_hash]

        try:

            base64_image = self.encode_image(
                image
            )

            prompt = f"""
Analyze this cropped object image.

The object detector classified it as:
"{yolo_class_name}"

Your task:

1. Identify the product as accurately as possible.
2. If the exact product cannot be identified,
   provide the most likely product category.
3. Provide an estimated typical market price only
   when reasonably possible.
4. Do not claim the price is a current retailer price.
5. Do not invent an exact model if the image
   does not provide enough evidence.

Return ONLY this format:

Product Name: <name>, Estimated Price: <price or Unknown>
"""

            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        "data:image/jpeg;base64,"
                                        f"{base64_image}"
                                    )
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 100,
                "temperature": 0
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=config.API_TIMEOUT
            )

            response.raise_for_status()

            data = response.json()

            product_info = (
                data["choices"][0]
                ["message"]["content"]
                .strip()
            )

            # To Save result in cache
            if image_hash:
                self.image_cache[
                    image_hash
                ] = product_info

            return product_info

        except requests.exceptions.Timeout:

            logger.warning(
                "OpenAI request timed out."
            )

            return (
                f"Product Name: "
                f"{yolo_class_name}, "
                f"Estimated Price: Unknown"
            )

        except requests.exceptions.RequestException as error:

            logger.error(
                f"OpenAI API error: {error}"
            )

            return (
                f"Product Name: "
                f"{yolo_class_name}, "
                f"Estimated Price: Unknown"
            )

        except Exception as error:

            logger.exception(
                f"Product extraction failed: {error}"
            )

            return (
                f"Product Name: "
                f"{yolo_class_name}, "
                f"Estimated Price: Unknown"
            )


# TO TRACK STATE

@dataclass
class TrackState:

    track_id: int
    class_name: str

    # Number of frames in which object appeared
    age: int = 1

    # Number of API requests made for this track
    api_requests: int = 0

    # AI-generated information
    product_info: Optional[str] = None

    # Whether an API request is currently running
    processing: bool = False

    # Last frame where object was seen
    last_seen_frame: int = 0


# To LOAD YOLO MODEL

def load_yolo_model() -> Optional[YOLO]:

    try:

        model = YOLO(
            config.MODEL_PATH
        )

        logger.info(
            "YOLO model loaded successfully."
        )

        return model

    except Exception as error:

        logger.exception(
            f"Failed to load YOLO model: {error}"
        )

        return None


# To SAFE IMAGE CROP

def crop_object(
    frame,
    box: Tuple[int, int, int, int]
):

    x1, y1, x2, y2 = box

    frame_height, frame_width = (
        frame.shape[:2]
    )

    # To Keep coordinates inside image boundaries
    x1 = max(0, min(x1, frame_width))
    x2 = max(0, min(x2, frame_width))

    y1 = max(0, min(y1, frame_height))
    y2 = max(0, min(y2, frame_height))

    # For Invalid bounding box
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    # For Empty image
    if crop.size == 0:
        return None

    crop_height, crop_width = (
        crop.shape[:2]
    )

    # To Ignore very small detections
    if (
        crop_height < config.MIN_CROP_SIZE
        or crop_width < config.MIN_CROP_SIZE
    ):
        return None

    return crop


# TO DRAW LABEL

def draw_detection(
    frame,
    box: Tuple[int, int, int, int],
    label: str
):

    x1, y1, x2, y2 = box

    color = (0, 255, 0)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    # To Limit label length
    if len(label) > 120:
        label = label[:117] + "..."

    text_y = max(y1 - 10, 20)

    cv2.putText(
        frame,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        color,
        1,
        cv2.LINE_AA
    )


# TO CLEAN OLD TRACKS

def cleanup_old_tracks(
    track_states: Dict[int, TrackState],
    current_frame: int,
    max_missing_frames: int = 100
):

    tracks_to_remove = []

    for track_id, state in track_states.items():

        if (
            current_frame
            - state.last_seen_frame
            > max_missing_frames
        ):
            tracks_to_remove.append(
                track_id
            )

    for track_id in tracks_to_remove:

        del track_states[track_id]

        logger.info(
            f"Removed inactive track "
            f"{track_id}"
        )


# TO PROCESS VIDEO

def process_video(
    video_path: str,
    output_filename: str
) -> str:

    # To Load YOLO

    model = load_yolo_model()

    if model is None:

        raise RuntimeError(
            "YOLO model could not be loaded."
        )

    # To Open video

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise FileNotFoundError(
            f"Could not open video: "
            f"{video_path}"
        )

    # Video properties

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    frame_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30

    # To Create output directory

    os.makedirs(
        config.UPLOAD_FOLDER,
        exist_ok=True
    )

    processed_video_path = os.path.join(
        config.UPLOAD_FOLDER,
        f"processed_{output_filename}"
    )

    # Video writer

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    video_writer = cv2.VideoWriter(
        processed_video_path,
        fourcc,
        fps,
        (
            frame_width,
            frame_height
        )
    )

    if not video_writer.isOpened():

        cap.release()

        raise RuntimeError(
            "Could not create output video."
        )

    # To Initialize AI product extractor

    product_extractor = ProductExtractor()

    # Track states

    track_states: Dict[
        int,
        TrackState
    ] = {}

    # Future -> Track ID mapping
    pending_futures: Dict[
        Future,
        int
    ] = {}

    current_frame = 0

    logger.info(
        f"Processing video with "
        f"{total_frames} frames."
    )

    
    # THREAD POOL

    executor = ThreadPoolExecutor(
        max_workers=config.MAX_WORKERS
    )

    try:

        # VIDEO LOOP

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            current_frame += 1

            # YOLO OBJECT TRACKING
            
            # persist=True maintains object IDs between frames

            results = model.track(
                frame,
                persist=True,
                tracker=config.TRACKER_CONFIG,
                conf=config.CONFIDENCE_THRESHOLD,
                verbose=False
            )

            # TO CHECK COMPLETED API REQUESTS
            
            # To avoid blocking the video loop.

            completed_futures = []

            for future, track_id in pending_futures.items():

                if future.done():

                    completed_futures.append(
                        future
                    )

                    try:

                        product_info = (
                            future.result()
                        )

                        if (
                            track_id
                            in track_states
                        ):

                            track_states[
                                track_id
                            ].product_info = (
                                product_info
                            )

                            track_states[
                                track_id
                            ].processing = False

                            logger.info(
                                f"Track {track_id} "
                                f"identified as: "
                                f"{product_info}"
                            )

                    except Exception as error:

                        logger.error(
                            f"Track {track_id} "
                            f"processing failed: "
                            f"{error}"
                        )

                        if (
                            track_id
                            in track_states
                        ):

                            track_states[
                                track_id
                            ].processing = False

            # To Remove completed futures
            for future in completed_futures:

                del pending_futures[
                    future
                ]

            # PROCESS YOLO RESULTS

            for result in results:

                # If tracker found no objects
                if (
                    result.boxes is None
                    or len(result.boxes) == 0
                ):
                    continue

                # To Tracking IDs may not exist initially
                if result.boxes.id is None:
                    continue

                boxes = (
                    result.boxes.xyxy
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                track_ids = (
                    result.boxes.id
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                class_ids = (
                    result.boxes.cls
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                confidences = (
                    result.boxes.conf
                    .cpu()
                    .numpy()
                )

                # TO PROCESS EVERY TRACKED OBJECT

                for (
                    box,
                    track_id,
                    class_id,
                    confidence
                ) in zip(
                    boxes,
                    track_ids,
                    class_ids,
                    confidences
                ):

                    x1, y1, x2, y2 = box.tolist()

                    box_tuple = (
                        x1,
                        y1,
                        x2,
                        y2
                    )

                    class_name = (
                        model.names[
                            int(class_id)
                        ]
                    )

                    # TO CREATE OR UPDATE TRACK STATE

                    if track_id not in track_states:

                        track_states[
                            track_id
                        ] = TrackState(
                            track_id=track_id,
                            class_name=class_name,
                            age=1,
                            last_seen_frame=current_frame
                        )

                    else:

                        state = track_states[
                            track_id
                        ]

                        state.age += 1

                        state.last_seen_frame = (
                            current_frame
                        )

                    state = track_states[
                        track_id
                    ]

                    # TO GET OBJECT CROP

                    crop = crop_object(
                        frame,
                        box_tuple
                    )

                    # SEND ONLY ONE API REQUEST
                    # PER TRACK
                    #
                    # Conditions:
                    # - Object has existed long enough
                    # - Valid crop exists
                    # - No existing result
                    # - No request currently running
                    # - Request limit not exceeded

                    should_analyze = (
                        state.age >= config.MIN_TRACK_AGE
                        and crop is not None
                        and state.product_info is None
                        and not state.processing
                        and (
                            state.api_requests
                            < config.MAX_API_REQUESTS_PER_TRACK
                        )
                    )

                    if should_analyze:

                        future = executor.submit(
                            product_extractor.extract_product,
                            crop.copy(),
                            class_name
                        )

                        pending_futures[
                            future
                        ] = track_id

                        state.processing = True

                        state.api_requests += 1

                        logger.info(
                            f"Submitted API request "
                            f"for Track {track_id} "
                            f"({class_name})"
                        )

                    # TO CREATE DISPLAY LABEL

                    if state.product_info:

                        label = (
                            f"ID {track_id} | "
                            f"{state.product_info}"
                        )

                    elif state.processing:

                        label = (
                            f"ID {track_id} | "
                            f"Analyzing..."
                        )

                    else:

                        label = (
                            f"ID {track_id} | "
                            f"{class_name} "
                            f"({confidence:.2f})"
                        )

                    # TO DRAW RESULT

                    draw_detection(
                        frame,
                        box_tuple,
                        label
                    )

            # TO CLEAN INACTIVE TRACKS

            cleanup_old_tracks(
                track_states,
                current_frame
            )

            # TO WRITE OUTPUT FRAME

            video_writer.write(
                frame
            )

            # PROGRESS

            if total_frames > 0:

                progress = (
                    current_frame
                    / total_frames
                ) * 100

                print(
                    f"Processing: "
                    f"{progress:.2f}% complete",
                    end="\r"
                )

    finally:

        # WAIT FOR BACKGROUND TASKS

        logger.info(
            "Finishing pending API requests..."
        )

        executor.shutdown(
            wait=True
        )

        
        #  TO RELEASE VIDEO RESOURCES

        cap.release()

        video_writer.release()

    logger.info(
        f"Processed video saved at: "
        f"{processed_video_path}"
    )

    return processed_video_path


# ============================================================
# FOR EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":

    input_video = "uploads/input_video.mp4"

    output_video_name = "output_video.mp4"

    try:

        output_path = process_video(
            input_video,
            output_video_name
        )

        print(
            "\nVideo processing completed "
            "successfully!"
        )

        print(
            f"Output: {output_path}"
        )

    except Exception as error:

        logger.exception(
            f"Video processing failed: {error}"
        )
