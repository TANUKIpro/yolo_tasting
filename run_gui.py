#!/usr/bin/env python3
"""
YOLO Tasting GUI Entry Point

Launch the Gradio-based GUI for interactive YOLO model testing.

Usage:
    python run_gui.py [options]

Options:
    --share         Create a public shareable link
    --host HOST     Server hostname (default: 127.0.0.1)
    --port PORT     Server port (default: 7860)
    --debug         Enable debug mode

Examples:
    # Basic launch (localhost only)
    python run_gui.py

    # Share publicly
    python run_gui.py --share

    # Custom host/port
    python run_gui.py --host 0.0.0.0 --port 8080

    # Debug mode
    python run_gui.py --debug
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        import gradio
    except ImportError:
        missing.append("gradio")

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        from ultralytics import YOLO
    except ImportError:
        missing.append("ultralytics")

    if missing:
        print("Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="YOLO Tasting GUI - Interactive YOLO model testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server hostname (default: 127.0.0.1)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port (default: 7860)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Print startup info
    print("=" * 60)
    print("  YOLO Tasting GUI")
    print("=" * 60)
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Share: {args.share}")
    print(f"  Debug: {args.debug}")
    print("=" * 60)
    print()

    # Import and launch
    from gui import launch_app

    launch_app(
        share=args.share,
        server_name=args.host,
        server_port=args.port,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
