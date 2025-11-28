import torch

_torch_load_original = torch.load.__wrapped__ if hasattr(torch.load, '__wrapped__') else torch.load

def _safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _torch_load_original(*args, **kwargs)

torch.load = _safe_torch_load
print("PyTorch load patched for weights_only=False")

import yolov5
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
from yolov5.utils.augmentations import letterbox
from yolov5.utils.plots import colors, Annotator
from yolov5.utils.torch_utils import select_device

import cv2
import numpy as np
from pathlib import Path
import os
import time
import matplotlib.pyplot as plt


print(f"YOLOv5 version: {yolov5.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")


DRIVE_BASE = "/home/ryo/workspace/yolo_tasting"
INPUT_DIR = f"{DRIVE_BASE}/common/input"
OUTPUT_DIR = f"{DRIVE_BASE}/yolo_v5/output"
WEIGHTS_DIR = f"{DRIVE_BASE}/yolo_v5/weights"

CONF_THRESHOLD = 0.5   # 信頼度閾値
IOU_THRESHOLD = 0.45   # NMS IoU閾値
IMG_SIZE = [640, 480]  # 推論サイズ [width, height]
MAX_DET = 1000         # 最大検出数

WEIGHTS_PATH = f"{WEIGHTS_DIR}/ycb.pt"
# オプション2: 標準COCO学習済みモデル (ダウンロード)
# !wget -P {WEIGHTS_DIR} https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
# WEIGHTS_PATH = f"{WEIGHTS_DIR}/yolov5s.pt"

# 動画処理設定 (メモリ節約用)
VIDEO_SAVE_INTERVAL = 30  # N フレームごとに書き出し


class YOLOv5Detector:
    def __init__(self, weights, imgsz=[640, 480], conf_thres=0.5, iou_thres=0.45,
                 max_det=1000, device='', half=False):
        """
        Args:
            weights: 重みファイルパス (.pt)
            imgsz: 推論サイズ [width, height]
            conf_thres: 信頼度閾値
            iou_thres: NMS IoU閾値
            max_det: 最大検出数
            device: デバイス ('', '0', 'cpu')
            half: FP16使用
        """
        self.device = select_device(device)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.half = half and self.device.type != 'cpu'

        # モデルロード (TidBots HSR と同じ方式)
        print(f"Loading model: {weights}")
        self.model = attempt_load(weights, device=self.device)
        self.stride = int(self.model.stride.max())
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        if self.half:
            self.model.half()

        self.imgsz = check_img_size(imgsz[0], s=self.stride)
        self.imgsz = imgsz  # TidBots HSR では固定値を使用

        print(f"Model loaded successfully")
        print(f"  Device: {self.device}")
        print(f"  Stride: {self.stride}")
        print(f"  Image size: {self.imgsz}")
        print(f"  Classes ({len(self.names)}): {list(self.names.values()) if isinstance(self.names, dict) else self.names}")

    @torch.no_grad()
    def detect(self, img_input, augment=False):
        """
        物体検出を実行

        Args:
            img_input: 入力画像 (numpy array BGR または ファイルパス)
            augment: TTA (Test Time Augmentation)

        Returns:
            detections: list of dict (x_min, y_min, x_max, y_max, confidence, class_name)
            annotated_img: 検出結果描画済み画像
        """
        # 画像読み込み
        if isinstance(img_input, str):
            img0 = cv2.imread(img_input)
        else:
            img0 = img_input.copy()

        # 前処理 (letterbox)
        img = letterbox(img0, self.imgsz, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        # Tensor変換
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img = img / 255.0
        if len(img.shape) == 3:
            img = img[None]  # batch dim追加

        # 推論
        pred = self.model(img, augment=augment, visualize=False)[0]

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                   None, False, max_det=self.max_det)

        # 結果処理
        detections = []
        annotator = Annotator(img0.copy(), line_width=2, example=str(self.names))

        for det in pred:
            if len(det):
                # スケール変換
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    class_name = self.names[c] if isinstance(self.names, list) else self.names.get(c, str(c))
                    label = f'{class_name} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

                    detections.append({
                        'x_min': int(xyxy[0]),
                        'y_min': int(xyxy[1]),
                        'x_max': int(xyxy[2]),
                        'y_max': int(xyxy[3]),
                        'confidence': float(conf),
                        'class_name': class_name
                    })

        return detections, annotator.result()

class VideoProcessorStreaming:
    """
    動画処理クラス (ストリーミング書き込み版)
    メモリを最小限に抑えつつ、連続的にDriveへ書き出し
    """

    def __init__(self, detector, output_dir):
        self.detector = detector
        self.output_dir = output_dir

    def process_video(self, video_path, output_name=None,
                      flush_interval=30, show_progress=True,
                      max_frames=None, skip_frames=0):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_fps = fps // (skip_frames + 1) if skip_frames > 0 else fps

        if max_frames:
            total_frames = min(total_frames, max_frames * (skip_frames + 1))

        print(f"Input: {video_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps} -> {output_fps} (output)")
        print(f"  Total frames: {total_frames}")

        if output_name is None:
            input_name = Path(video_path).stem
            output_name = f"{input_name}_detected.mp4"
        output_path = f"{self.output_dir}/{output_name}"

        # ローカル一時ファイル (高速書き込み)
        temp_path = "/tmp/temp_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, output_fps, (width, height))

        frame_count = 0
        processed_count = 0
        start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                if skip_frames > 0 and (frame_count - 1) % (skip_frames + 1) != 0:
                    continue

                if max_frames and processed_count >= max_frames:
                    break

                _, annotated = self.detector.detect(frame)
                out.write(annotated)
                processed_count += 1

                # 進捗表示のみ (VideoWriterはrelease しない)
                if processed_count % flush_interval == 0:
                    if show_progress:
                        elapsed = time.time() - start_time
                        fps_actual = processed_count / elapsed
                        remaining = (total_frames // (skip_frames + 1) - processed_count)
                        eta = remaining / fps_actual if fps_actual > 0 else 0
                        print(f"Progress: {processed_count} frames processed")
                        print(f"Speed: {fps_actual:.1f} fps | ETA: {eta:.0f}s")

                if processed_count % 50 == 0:
                    torch.cuda.empty_cache()

        finally:
            cap.release()
            out.release()

            # 処理完了後にDriveへコピー
            import shutil
            if os.path.exists(temp_path):
                print(f"Copying to Drive: {output_path}")
                shutil.copy2(temp_path, output_path)
                os.remove(temp_path)

        total_time = time.time() - start_time
        print(f"\nCompleted: {processed_count} frames in {total_time:.1f}s")
        print(f"Average speed: {processed_count/total_time:.1f} fps")
        print(f"Output saved: {output_path}")

        return processed_count


def detect_and_save_image(image_path, output_path=None, show=True):
    """
    単一画像に対して物体検出を実行

    Args:
        image_path: 入力画像パス
        output_path: 出力画像パス (Noneの場合は自動生成)
        show: 結果を表示するか

    Returns:
        検出結果のリスト
    """
    # 推論実行
    detections, annotated_img = detector.detect(image_path)

    # 結果保存
    if output_path is None:
        input_name = Path(image_path).stem
        output_path = f"{OUTPUT_DIR}/{input_name}_detected.jpg"

    cv2.imwrite(output_path, annotated_img)
    print(f"Result saved to: {output_path}")

    # 結果表示
    if show:
        # BGR to RGB for display
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f'Detections: {len(detections)}')
        plt.show()

    # 検出結果表示
    print(f"\nDetections ({len(detections)}):")
    for i, d in enumerate(detections):
        print(f"  [{i}] {d['class_name']}: {d['confidence']:.2f} @ ({d['x_min']}, {d['y_min']}) - ({d['x_max']}, {d['y_max']})")

    return detections

def batch_process_images(input_dir, output_dir):
    """ディレクトリ内の全画像を処理"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG']
    images = [f for f in os.listdir(input_dir)
              if Path(f).suffix.lower() in image_extensions]

    print(f"Processing {len(images)} images...")

    for i, img_name in enumerate(images):
        img_path = f"{input_dir}/{img_name}"
        output_path = f"{output_dir}/{Path(img_name).stem}_detected.jpg"

        _, annotated = detector.detect(img_path)
        cv2.imwrite(output_path, annotated)

        print(f"[{i+1}/{len(images)}] {img_name} -> {output_path}")

        # メモリ解放
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()

    print(f"\nBatch processing complete. Results in: {output_dir}")


if __name__ == "__main__":
    # 検出器初期化
    if os.path.exists(WEIGHTS_PATH):
        detector = YOLOv5Detector(
            weights=WEIGHTS_PATH,
            imgsz=IMG_SIZE,
            conf_thres=CONF_THRESHOLD,
            iou_thres=IOU_THRESHOLD,
            max_det=MAX_DET
        )
    else:
        print(f"ERROR: Weights file not found: {WEIGHTS_PATH}")
        print(f"Please upload your weights to: {WEIGHTS_DIR}")
        print("\nOr download pretrained weights:")
        print(f"  !wget -P {WEIGHTS_DIR} https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt")

    # 単一画像推論テスト
    is_image_test = False
    if is_image_test:
        # テスト画像の一覧
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG']
        images = [f for f in os.listdir(INPUT_DIR)
                if Path(f).suffix.lower() in image_extensions]

        if images:
            print(f"Found {len(images)} images in {INPUT_DIR}:")
            for img in images:
                print(f"  - {img}")

            # 最初の画像で推論テスト
            test_image = f"{INPUT_DIR}/{images[0]}"
            print(f"\nTesting with: {test_image}")
            detect_and_save_image(test_image)
    
    # バッチ画像処理テスト
    # バッチ処理実行
    is_batch_image_test = True
    if is_batch_image_test:
        batch_process_images(INPUT_DIR, OUTPUT_DIR)

    # 単一動画推論テスト
    is_video_test = False
    if is_video_test:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        videos = [f for f in os.listdir(INPUT_DIR)
                if Path(f).suffix.lower() in video_extensions]

        if videos:
            print(f"Found {len(videos)} videos in {INPUT_DIR}:")
            for vid in videos:
                print(f"  - {vid}")

            # 最初の動画で推論テスト
            test_video = f"{INPUT_DIR}/{videos[0]}"
            print(f"\nProcessing: {test_video}")

            processor = VideoProcessorStreaming(detector, OUTPUT_DIR)
            processor.process_video(
                test_video,
                flush_interval=30,  # 30フレームごとに保存
                skip_frames=0,      # 全フレーム処理 (1=1フレームおき, 2=2フレームおき...)
                max_frames=None     # 全フレーム処理 (テスト時は100などに設定)
            )
        else:
            print(f"No videos found in {INPUT_DIR}")
            print("Please upload videos to this directory and re-run this cell.")
