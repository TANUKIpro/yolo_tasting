# ROS YOLO - HSR互換 YOLOv5 物体認識環境

tidbots_docker/hsr-tid/docker-yolov5-rosと互換性のあるROS1 Noetic + Docker環境。
YCBデータセットで学習済みのYOLOv5モデルを使用した物体認識を行う。

## ディレクトリ構造

```
ros_yolo/
├── docker/
│   ├── Dockerfile              # ROS Noetic + CUDA + PyTorch + YOLOv5
│   ├── Dockerfile.camera       # カメラ用軽量イメージ
│   ├── docker-compose.yml      # 3サービス構成
│   ├── ros_entrypoint.sh       # ROS環境セットアップ
│   └── requirements.txt        # Python依存関係
├── catkin_ws/
│   └── src/
│       ├── yolov5_ros/         # YOLOv5 ROSパッケージ
│       │   ├── scripts/
│       │   │   ├── ros_yolov5.py   # YOLOv5検出ノード
│       │   │   └── viewer.py       # 検出結果ビューアー
│       │   ├── msg/
│       │   │   ├── RecognitionObject.msg
│       │   │   └── RecognitionObjectArray.msg
│       │   └── weights/
│       │       └── ycb.pt          # YCB学習済みモデル
│       └── camera_publisher/   # カメラパブリッシャー
│           └── launch/
│               └── camera.launch
└── README.md
```

## クイックスタート（PC完結型テスト環境）

HSR実機がなくても、PC1台で完結するテスト環境を提供。
ノートPCカメラからHSR互換トピックを発行し、YOLOv5で物体認識を行う。

### アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                        PC (Docker)                          │
│                                                             │
│  ┌─────────────┐    ROS Topics     ┌─────────────┐         │
│  │   camera    │ ─────────────────>│    yolo     │         │
│  │  container  │  /hsrb/head_rgbd  │  container  │         │
│  │             │  sensor/rgb/      │             │         │
│  │  usb_cam    │  image_rect_color │ ros_yolov5  │         │
│  └─────────────┘                   └──────┬──────┘         │
│        │                                  │                 │
│        │ /dev/video0              /yolov5_obj               │
│        ▼                                  │                 │
│  ┌─────────────┐                          ▼                 │
│  │  PC Camera  │                   ┌─────────────┐         │
│  └─────────────┘                   │   viewer    │         │
│                                    │  (画像+検出 │         │
│                                    │   結果表示) │         │
│                                    └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### 1. Dockerイメージのビルド

```bash
cd ros_yolo/docker
docker-compose build
```

### 2. 全サービス起動

```bash
docker-compose up -d
```

3つのサービスが起動：
- **roscore**: ROS Master
- **camera**: PCカメラからHSR互換トピックを発行
- **yolo**: YOLOv5物体検出器

### 3. ビューアー起動（検出結果確認）

```bash
# X11表示を許可（初回のみ）
xhost +local:docker

# ビューアー起動
docker-compose exec yolo python3 /catkin_ws/src/yolov5_ros/scripts/viewer.py
```

ビューアー操作：
- `q`キー: 終了

### 4. 停止

```bash
docker-compose down
```

## ROSトピック

| トピック | 発行者 | 購読者 | 型 |
|---------|--------|--------|-----|
| `/hsrb/head_rgbd_sensor/rgb/image_rect_color` | camera | yolo, viewer | sensor_msgs/Image |
| `/yolov5_obj` | yolo | viewer | RecognitionObjectArray |

## HSR実機接続モード

### 環境変数の設定

```bash
export ROS_MASTER_URI=http://hsrb.local:11311
export ROS_IP=192.168.1.xxx  # 自分のIPアドレス
```

### 検出ノードのみ起動

```bash
docker-compose run yolo python3 /catkin_ws/src/yolov5_ros/scripts/ros_yolov5.py \
    --weights /catkin_ws/src/yolov5_ros/weights/ycb.pt \
    --topic /hsrb/head_rgbd_sensor/rgb/image_rect_color \
    --view_img
```

## コマンドライン引数

| 引数 | デフォルト値 | 説明 |
|------|-------------|------|
| `--weights` | `yolov5s.pt` | モデルファイルパス |
| `--topic` | `/camera/rgb/image_rect_color` | 入力画像トピック |
| `--conf-thres` | `0.5` | 信頼度閾値 |
| `--iou-thres` | `0.45` | NMS IoU閾値 |
| `--imgsz` | `640` | 入力画像サイズ |
| `--max-det` | `1000` | 最大検出数 |
| `--device` | `` (自動) | 使用デバイス (cuda/cpu) |
| `--view_img` | `false` | 検出結果を表示 |
| `--half` | `false` | FP16推論 |

## カメラ設定

カメラデバイスや解像度を変更する場合：

```bash
# 異なるカメラデバイスを使用
docker-compose run camera roslaunch camera_publisher camera.launch video_device:=/dev/video1

# 解像度変更
docker-compose run camera roslaunch camera_publisher camera.launch image_width:=1280 image_height:=720
```

## カスタムメッセージ

### RecognitionObject.msg
```
int16 x_min
int16 y_min
int16 x_max
int16 y_max
float32 confidence
string class_name
```

### RecognitionObjectArray.msg
```
std_msgs/Header header
yolov5_ros/RecognitionObject[] array
```

## 検出結果の確認

```bash
# 検出結果をモニター
docker-compose exec yolo rostopic echo /yolov5_obj

# 画像トピック確認
docker-compose exec yolo rostopic hz /hsrb/head_rgbd_sensor/rgb/image_rect_color
```

## YOLOv5 API特徴

このコードでは `pip install yolov5` パッケージを使用：

- `attempt_load()` でカスタムモデルをロード
- 手動で `letterbox()` 前処理
- 手動で `non_max_suppression()` を呼び出し
- オフライン動作可能（torch.hubと異なりネットワーク不要）

## トラブルシューティング

### GPUが認識されない
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi
```

### X11表示エラー
```bash
xhost +local:docker
```

### カメラが認識されない
```bash
# 利用可能なカメラデバイス確認
ls -la /dev/video*

# カメラ情報確認
v4l2-ctl --list-devices
```

### catkin_makeエラー
```bash
# コンテナ内で
cd /catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

## 次のステップ

- YOLOv8対応の追加（ros_yolov8.pyを新規作成予定）
- YOLOv5 vs YOLOv8の性能比較機能
