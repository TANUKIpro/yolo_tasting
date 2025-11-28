# YOLO Tasting

YOLOv5とYOLOv11の物体検出性能を比較・評価するPythonテストフレームワーク。

## 技術スタック

- **言語**: Python 3
- **主要ライブラリ**: PyTorch, Ultralytics, OpenCV, NumPy
- **対応モデル**: YOLOv5 (YCB), YOLOv11 (nano/small/medium/large), YOLO-World, Pose, Segmentation

## ディレクトリ構造

```
yolo_tasting/
├── common/                 # 共通モジュール
│   ├── __init__.py
│   ├── config.py           # 共通設定（パス、閾値、カスタムクラス）
│   ├── utils.py            # ユーティリティ関数
│   └── video_processor.py  # 統合動画処理クラス
├── detectors/              # 検出器モジュール
│   ├── __init__.py
│   ├── base.py             # 抽象基底クラス (BaseDetector)
│   ├── yolov5_detector.py  # YOLOv5検出器
│   └── yolov11_detector.py # YOLOv11検出器
├── gui/                    # GUIモジュール（Gradio）
│   ├── __init__.py
│   ├── app.py              # Gradioアプリケーション
│   ├── model_registry.py   # モデル定義・レジストリ
│   └── detector_manager.py # 検出器マネージャー
├── common/input/           # テスト用入力データ（画像・動画）
├── yolo_v5/output/         # YOLOv5検出結果
├── yolo_v11/output/        # YOLOv11検出結果
├── yolo_test_v5_ybc.py     # YOLOv5テストスクリプト
├── yolo_test_v11_ultralytics.py  # YOLOv11テストスクリプト
└── run_gui.py              # GUIエントリーポイント
```

## 主要モジュール

### common/config.py
- `Config`: 共通設定クラス（ディレクトリパス、検出パラメータ、カスタムクラスプリセット）

### common/utils.py
- `get_image_files()`: ディレクトリ内の画像ファイル一覧取得
- `get_video_files()`: ディレクトリ内の動画ファイル一覧取得
- `print_detection_results()`: 検出結果の整形出力
- `show_detection_image()`: 検出結果の画像表示

### common/video_processor.py
- `VideoProcessor`: 統合動画処理クラス（メモリ効率化、GPU管理）

### detectors/base.py
- `BaseDetector`: 検出器の抽象基底クラス（共通インターフェース）

### detectors/yolov5_detector.py
- `YOLOv5Detector`: YOLOv5物体検出器

### detectors/yolov11_detector.py
- `YOLOv11Detector`: YOLOv11/YOLO-World統合検出器（物体検出、ポーズ推定、セグメンテーション）

### gui/model_registry.py
- `ModelRegistry`: 全34モデルの定義（YOLOv5/v11/World、各タスク・サイズ）
- `get_versions()`: 利用可能なYOLOバージョン一覧
- `get_tasks_for_version()`: バージョン別タスク一覧
- `get_model_key()`: モデルキー生成

### gui/detector_manager.py
- `DetectorManager`: 検出器のロード・キャッシュ・パラメータ管理
- `get_detector_manager()`: グローバルマネージャー取得

### gui/app.py
- `YOLOTastingApp`: Gradioアプリケーションクラス
- `create_app()`: Gradio Blocksアプリ生成
- `launch_app()`: GUIサーバー起動

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

## GUI使用方法

### 起動

```bash
# 基本起動（localhost:7860）
python run_gui.py

# 公開リンク生成
python run_gui.py --share

# カスタムホスト/ポート
python run_gui.py --host 0.0.0.0 --port 8080
```

### 機能

1. **モデル選択**: YOLOv5/v11/Worldをドロップダウンで選択
2. **タスク選択**: Detection, Pose, Segmentation, Classification, OBB
3. **サイズ選択**: Nano, Small, Medium, Large, XLarge
4. **パラメータ調整**: スライダーでConf/IoU/ImgSize/MaxDet設定
5. **リアルタイム更新**: パラメータ変更時に自動再検出
6. **カスタムクラス**: YOLO-World用のプリセット（RoboCup/YCB等）

### 対応モデル（34種類）

| バージョン | タスク | サイズ |
|-----------|--------|--------|
| YOLOv5 | Detection | n/s/m/l/x |
| YOLOv11 | Detection | n/s/m/l/x |
| YOLOv11 | Pose | n/s/m/l/x |
| YOLOv11 | Segmentation | n/s/m/l/x |
| YOLOv11 | Classification | n/s/m/l/x |
| YOLOv11 | OBB | n/s/m/l/x |
| YOLO-World | Detection | s/m/l/x |

## 開発メモ

- 動画処理時は一時ファイルとGPUキャッシュ管理を使用
- YOLO-WorldではRoboCup@Home/YCB用のプリセットクラスセットあり
- フレームスキップオプションで処理速度調整可能
- 検出器は`BaseDetector`を継承し、`detect()`メソッドを実装
- GUIはGradio 6.0+を使用（pip install gradio）
