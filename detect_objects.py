"""Detect objects in one image and save a labeled copy alongside it."""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to the input image")
    parser.add_argument("--model", default="yolo26n.pt", help="YOLO detection weights (.pt)")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image file does not exist: {args.image}")

    model = YOLO(args.model)
    result = model.predict(source=str(args.image), conf=0.25, verbose=False)[0]

    # Each detection includes its class name, confidence, and box in pixel coordinates.
    print(json.dumps(result.summary(), indent=2))

    output = args.image.with_name(f"{args.image.stem}_detected.jpg")
    result.save(filename=str(output))


if __name__ == "__main__":
    main()
