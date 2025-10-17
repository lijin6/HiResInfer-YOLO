# HiResInfer-YOLO

**HiResInfer-YOLO** is a high-resolution dam crack detection project, **based on YOLOv12 and the Ultralytics framework**. This project leverages deep learning for accurate and efficient detection of cracks on dam surfaces.

We gratefully acknowledge the excellent code and frameworks provided by [Ultralytics](https://github.com/ultralytics/ultralytics) and [SAHI](https://github.com/obss/sahi), which greatly facilitated our development.

---

## Features

* Based on **YOLOv12** with modifications for crack detection
* Built on the **Ultralytics** framework for easy training and inference
* Supports high-resolution image input
* Provides fast and accurate crack detection

---

## Installation

1. Clone the repository:

```bash
git clone git@github.com:yourusername/HiResInfer-YOLO.git
cd HiResInfer-YOLO
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies mainly include:

* Python 3.8+
* torch
* torchvision
* opencv-python
* numpy
* ultralytics
* sahi

---

## Usage

### Training

```bash
python train.py --data data/ --epochs 50 --img-size 640 --batch-size 8
```

### Detection / Inference

```bash
python detect.py --weights yolov12_modified.pt --source data/images/ --output results/
```

* `--weights` : Trained model weights
* `--source`  : Image or folder to detect
* `--output`  : Path to save results

---

## Citation

If you use this project in your research, please consider citing:

```bibtex
@article{tian2025yolov12,
  title={YOLOv12: Attention-Centric Real-Time Object Detectors},
  author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
  journal={arXiv preprint arXiv:2502.12524},
  year={2025}
}
```

---

## Acknowledgements

We sincerely thank the developers of:

* [Ultralytics](https://github.com/ultralytics/ultralytics) – for YOLOv12 framework and code
* [SAHI](https://github.com/obss/sahi) – for enhanced small object detection utilities

Their excellent code greatly facilitated the development of this project.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---