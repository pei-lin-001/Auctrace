from __future__ import annotations

import base64
from typing import Any

from PIL import Image


def encode_image_to_base64(image_path: str) -> str:
    """Convert an image to base64 string (RGB JPEG)."""

    with Image.open(image_path) as img:
        if img.mode == "RGBA":
            img = img.convert("RGB")

        import io

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

    return base64.b64encode(image_bytes).decode("utf-8")


def build_vision_content(
    text: str,
    image_paths: str | list[str],
    *,
    max_images: int = 25,
    image_detail: str = "low",
) -> list[dict[str, Any]]:
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for image_path in image_paths[:max_images]:
        base64_image = encode_image_to_base64(image_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": image_detail,
                },
            }
        )
    return content

