import ultralytics
from ultralytics import YOLO

import torch
import cv2
import numpy as np
from pathlib import Path
import os
import time
import matplotlib.pyplot as plt


print(f"Ultralytics version: {ultralytics.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# 環境チェック
ultralytics.checks()


DRIVE_BASE = "/home/ryo/workspace/yolo_tasting"
INPUT_DIR = f"{DRIVE_BASE}/common/input"
OUTPUT_DIR = f"{DRIVE_BASE}/yolo_v11/output"
WEIGHTS_DIR = f"{DRIVE_BASE}/yolo_v11/weights"

# 推論設定
CONF_THRESHOLD = 0.5   # 信頼度閾値
IOU_THRESHOLD = 0.45   # NMS IoU閾値
IMG_SIZE = 640         # 推論サイズ (正方形)
MAX_DET = 300          # 最大検出数

# ↓ YOLO11 モデル (標準物体検出)
# モデル選択
MODEL_NAME = "yolo11m"  # yolo11n, yolo11s, yolo11m, yolo11l, yolo11x

# モデルロード
model_yolo11 = YOLO(f"{MODEL_NAME}.pt")

print(f"\nModel: {MODEL_NAME}")
print(f"Task: {model_yolo11.task}")
print(f"Classes: {len(model_yolo11.names)} classes (COCO)")

# YOLO11 クラス一覧 (COCO 80クラス)
# print("COCO Classes:")
# for idx, name in model_yolo11.names.items():
#     print(f"  {idx:2d}: {name}")

# ↓ YOLO-World モデル (ゼロショット検出)
# YOLO-Worldモデルのロード
model_world = YOLO("yolov8m-world.pt")
print(f"Model: YOLO-World")
print(f"Task: {model_world.task}")

# YOLO-World: カスタムクラスの設定
# ロボカップ@Home向けカスタムクラス
CUSTOM_CLASSES_ROBOCUP = [
    "person", "cup", "bottle", "bowl", "plate",
    "fork", "knife", "spoon", "banana", "apple",
    "orange", "sandwich", "book", "cell phone", "remote",
    "laptop", "chair", "couch", "dining table", "potted plant"
]

# YCBオブジェクト風カスタムクラス
CUSTOM_CLASSES_YCB = [
    "cracker box", "sugar box", "tomato soup can", "mustard bottle",
    "tuna fish can", "pudding box", "gelatin box", "potted meat can",
    "banana", "strawberry", "apple", "lemon", "peach", "pear", "orange", "plum",
    "pitcher", "bowl", "mug", "plate", "fork", "spoon", "knife", "spatula",
    "sponge", "power drill", "wood block", "scissors", "marker", "clamp",
    "tennis ball", "golf ball", "baseball", "dice", "rubiks cube"
]

# 例3: シンプルなカスタムクラス
CUSTOM_CLASSES_SIMPLE = [
    "person", "cup", "bottle", "book", "phone", "laptop"
]

# 使用するクラスを選択
ACTIVE_CLASSES = CUSTOM_CLASSES_ROBOCUP

# モデルにクラスを設定
model_world.set_classes(ACTIVE_CLASSES)

print(f"Active classes ({len(ACTIVE_CLASSES)}):")
for i, cls in enumerate(ACTIVE_CLASSES):
    print(f"  {i}: {cls}")


class UnifiedYOLODetector:
    """
    YOLO11 / YOLO-World 統合検出器
    """

    def __init__(self, model, conf=0.5, iou=0.45, imgsz=640, max_det=300):
        self.model = model
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det

    def detect(self, source, save=False, save_path=None):
        """
        物体検出を実行

        Args:
            source: 画像パス、numpy配列、またはURL
            save: 結果を保存するか
            save_path: 保存先パス

        Returns:
            detections: 検出結果リスト
            annotated_img: 描画済み画像
        """
        # 推論実行
        results = self.model(
            source,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            verbose=False
        )

        # 結果処理
        result = results[0]
        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]

                detections.append({
                    'x_min': int(xyxy[0]),
                    'y_min': int(xyxy[1]),
                    'x_max': int(xyxy[2]),
                    'y_max': int(xyxy[3]),
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': cls_name
                })

        # 描画済み画像
        annotated_img = result.plot()

        # 保存
        if save and save_path:
            cv2.imwrite(save_path, annotated_img)

        return detections, annotated_img

    def detect_and_show(self, source, figsize=(12, 8)):
        """検出して表示"""
        detections, annotated_img = self.detect(source)

        plt.figure(figsize=figsize)
        plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f'Detections: {len(detections)}')
        plt.show()

        print(f"\nDetections ({len(detections)}):")
        for i, d in enumerate(detections):
            print(f"  [{i}] {d['class_name']}: {d['confidence']:.2f}")

        return detections


class VideoProcessor:
    """
    動画処理クラス (メモリ効率化・ランタイム切断対策済み)
    """

    def __init__(self, detector, output_dir):
        self.detector = detector
        self.output_dir = output_dir

    def process_video(self, video_path, output_name=None,
                      show_progress=True, max_frames=None, skip_frames=0):
        """
        動画を処理

        Args:
            video_path: 入力動画パス
            output_name: 出力ファイル名
            show_progress: 進捗表示
            max_frames: 最大処理フレーム数
            skip_frames: スキップフレーム数 (高速化用)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # 動画情報
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

        # 出力設定
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

                # フレームスキップ
                if skip_frames > 0 and (frame_count - 1) % (skip_frames + 1) != 0:
                    continue

                if max_frames and processed_count >= max_frames:
                    break

                # 推論
                _, annotated = self.detector.detect(frame)
                out.write(annotated)
                processed_count += 1

                # 進捗表示
                if show_progress and processed_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = processed_count / elapsed
                    remaining = (total_frames // (skip_frames + 1) - processed_count)
                    eta = remaining / fps_actual if fps_actual > 0 else 0
                    print(f"Progress: {processed_count} frames processed")
                    print(f"Speed: {fps_actual:.1f} fps | ETA: {eta:.0f}s")

                # メモリ解放
                if processed_count % 50 == 0:
                    torch.cuda.empty_cache()

        finally:
            cap.release()
            out.release()

            # Driveへコピー
            import shutil
            if os.path.exists(temp_path):
                print(f"\nCopying to Drive: {output_path}")
                shutil.copy2(temp_path, output_path)
                os.remove(temp_path)

        total_time = time.time() - start_time
        print(f"\nCompleted: {processed_count} frames in {total_time:.1f}s")
        print(f"Average speed: {processed_count/total_time:.1f} fps")
        print(f"Output saved: {output_path}")

        return processed_count

if __name__ == "__main__":
    # 検出器インスタンス作成
    # YOLO11 検出器 (COCO 80クラス)
    detector_yolo11 = UnifiedYOLODetector(
        model_yolo11,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE
    )

    # YOLO-World 検出器 (カスタムクラス)
    detector_world = UnifiedYOLODetector(
        model_world,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE
    )

    # テスト画像の準備
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG']
    images = [f for f in os.listdir(INPUT_DIR)
            if Path(f).suffix.lower() in image_extensions]

    if images:
        print(f"Found {len(images)} images in {INPUT_DIR}:")
        for img in images:
            print(f"  - {img}")
        TEST_IMAGE = f"{INPUT_DIR}/{images[0]}"

    print(f"\nTest image: {TEST_IMAGE}")

    is_detect_yolo11 = False
    if is_detect_yolo11:
        # YOLO11 で検出
        print("=" * 50)
        print("YOLO11 Detection (COCO 80 classes)")
        print("=" * 50)
        detections_yolo11 = detector_yolo11.detect_and_show(TEST_IMAGE)

    is_detect_yolo_world = False
    if is_detect_yolo_world:
        # YOLO-World で検出 (カスタムクラス)
        print("=" * 50)
        print(f"YOLO-World Detection (Custom {len(ACTIVE_CLASSES)} classes)")
        print("=" * 50)
        detections_world = detector_world.detect_and_show(TEST_IMAGE)
    
    is_detect_yolo_world_custom = False
    if is_detect_yolo_world_custom:
        # YOLO-World で検出 (カスタムクラス)
        print("=" * 50)
        print(f"YOLO-World Detection (Custom {len(ACTIVE_CLASSES)} classes)")
        print("=" * 50)
        detections_world = detector_world.detect_and_show(TEST_IMAGE)

    is_yolo_world_dynamic_class_change = False
    if is_yolo_world_dynamic_class_change:
        # YOLO-World: クラスを動的に変更してテスト
        print("=" * 50)
        print("YOLO-World: Dynamic Class Change Demo")
        print("=" * 50)

        def set_world_classes_safe(model, classes):
            """
            YOLO-Worldのクラスを安全に変更
            デバイス不一致エラーを回避
            """
            original_device = next(model.model.parameters()).device
            model.to("cpu")
            model.set_classes(classes)
            model.to(original_device)

        # 特定のオブジェクトのみを検出
        specific_classes = ["cup", "tray"]
        set_world_classes_safe(model_world, specific_classes)

        print(f"\nSearching for: {specific_classes}")
        detector_world.detect_and_show(TEST_IMAGE)

        # 元のクラスに戻す
        set_world_classes_safe(model_world, ACTIVE_CLASSES)
    
    is_video_test = False
    if is_video_test:
        # 動画処理テスト
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        videos = [f for f in os.listdir(INPUT_DIR)
                if Path(f).suffix.lower() in video_extensions]

        if videos:
            print(f"Found {len(videos)} videos:")
            for vid in videos:
                print(f"  - {vid}")

            test_video = f"{INPUT_DIR}/{videos[0]}"
            print(f"\nProcessing with YOLO11: {test_video}")

            processor = VideoProcessor(detector_yolo11, OUTPUT_DIR)
            processor.process_video(
                test_video,
                skip_frames=0,
                max_frames=None
            )
        else:
            print(f"No videos found in {INPUT_DIR}")
        
    is_yolo11_segmentation_test = False
    if is_yolo11_segmentation_test:
        # YOLO11 セグメンテーションモデル
        model_seg = YOLO("yolo11l-seg.pt")

        print(f"Segmentation model loaded")
        print(f"Task: {model_seg.task}")

        # セグメンテーション実行
        results_seg = model_seg(TEST_IMAGE, conf=CONF_THRESHOLD)

        # 結果表示
        annotated_seg = results_seg[0].plot()
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(annotated_seg, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('YOLO11 Segmentation')
        plt.show()
    
    is_yolo11_pose_test = True
    if is_yolo11_pose_test:
        # YOLO11 姿勢推定モデル
        model_pose = YOLO("yolo11m-pose.pt")

        print(f"Pose estimation model loaded")
        print(f"Task: {model_pose.task}")

        # 姿勢推定実行
        results_pose = model_pose(f"{INPUT_DIR}/pose_sample.jpg", conf=CONF_THRESHOLD)

        # 結果表示
        annotated_pose = results_pose[0].plot()
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(annotated_pose, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('YOLO11 Pose Estimation')
        plt.show()
