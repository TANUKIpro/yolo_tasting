# YOLO Tasting

YOLOv5とYOLOv11の物体検出性能を比較・評価するPythonテストフレームワーク。

## 技術スタック

- **言語**: Python 3
- **主要ライブラリ**: PyTorch, Ultralytics, OpenCV, NumPy
- **対応モデル**: YOLOv5 (YCB), YOLOv11 (nano/small/medium/large), YOLO-World, Pose, Segmentation

## ディレクトリ構造

```
yolo_tasting/
├── common/input/          # テスト用入力データ（画像・動画）
├── yolo_v5/output/        # YOLOv5検出結果
├── yolo_v11/output/       # YOLOv11検出結果
├── yolo_test_v5_ybc.py    # YOLOv5テストスクリプト
└── yolo_test_v11_ultralytics.py  # YOLOv11テストスクリプト
```

## 主要ファイル

| ファイル | 役割 |
|---------|------|
| `yolo_test_v5_ybc.py` | YOLOv5検出器。`YOLOv5Detector`クラスと`VideoProcessorStreaming`クラスを実装 |
| `yolo_test_v11_ultralytics.py` | YOLOv11統合検出器。物体検出・ポーズ推定・セグメンテーション対応 |

## 主要クラス

### yolo_test_v5_ybc.py
- `YOLOv5Detector`: モデル読込・推論処理
- `VideoProcessorStreaming`: メモリ効率的な動画処理

### yolo_test_v11_ultralytics.py
- `UnifiedYOLODetector`: 統合検出インターフェース（COCO/YOLO-World/Pose/Segmentation）
- `VideoProcessor`: GPU管理付き動画処理

## 検出パラメータ（デフォルト値）

- 信頼度閾値: 0.5
- NMS IoU閾値: 0.45
- 画像サイズ: 640x640
- 最大検出数: 300-1000

## 対応タスク

1. **物体検出**: COCO 80クラス / カスタムクラス（YOLO-World）
2. **ポーズ推定**: 17点キーポイント検出
3. **セグメンテーション**: インスタンスセグメンテーション

## 入出力形式

- **入力**: JPG, PNG, MP4, AVI, MOV, MKV, WebM
- **出力**: バウンディングボックス付き画像・動画、検出メタデータ

## 開発メモ

- 動画処理時は一時ファイルとGPUキャッシュ管理を使用
- YOLO-WorldではRoboCup@Home/YCB用のプリセットクラスセットあり
- フレームスキップオプションで処理速度調整可能
