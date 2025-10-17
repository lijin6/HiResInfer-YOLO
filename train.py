from ultralytics import YOLO
from ultralytics import RTDETR
from ultralytics import NAS
# model = NAS("yolo_nas_s.pt")

model = YOLO("yolov12n.yaml").load('yolov12n.pt') 


results = model.train(
    data='crack.yaml',
    epochs=300,
    batch=32,
    imgsz=1280,
    patience=30,
    device=[0,1],
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5.0,
    translate=0.1,
    scale=0.9,
    shear=0.0,
    perspective=0.0005,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.3,
    erasing=0.4,
    auto_augment='randaugment',
    save_period=10,
    plots=True,
    rect=False,  # 禁用矩形训练
    overlap_mask=True,
    mask_ratio=4,
    dropout=0.1,
    val=True,
    amp=True  # 自动混合精度
)

# 验证配置
metrics = model.val(
    conf=0.001,  # 降低置信度阈值
    iou=0.5,     # 小目标适用IoU阈值
    max_det=500,  # 提高最大检测数
    half=True    # FP16验证
)