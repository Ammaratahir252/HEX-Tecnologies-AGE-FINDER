import os
import urllib.request

os.makedirs("models", exist_ok=True)

MODELS = {
    # ── Face Detector ─────────────────────────────────────────────────────────
    "models/deploy.prototxt": (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "samples/dnn/face_detector/deploy.prototxt"
    ),
    "models/res10_300x300_ssd_iter_140000.caffemodel": (
        "https://github.com/opencv/opencv_3rdparty/raw/"
        "dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    ),
    # ── Age Estimator ─────────────────────────────────────────────────────────
    "models/age_deploy.prototxt": (
        "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/"
        "master/age_deploy.prototxt"
    ),
    "models/age_net.caffemodel": (
        "https://drive.google.com/uc?export=download&id=1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW"
    ),
    # ── Gender Classifier ─────────────────────────────────────────────────────
    "models/gender_deploy.prototxt": (
        "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/"
        "master/gender_deploy.prototxt"
    ),
    "models/gender_net.caffemodel": (
        "https://drive.google.com/uc?export=download&id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ"
    ),
}


def download(path: str, url: str) -> None:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1_048_576
        print(f"  ✓ Already exists: {path}  ({size_mb:.1f} MB)")
        return
    print(f"  ↓ Downloading: {path}")
    try:
        urllib.request.urlretrieve(url, path)
        size_mb = os.path.getsize(path) / 1_048_576
        print(f"    Saved ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        print(f"    Please download manually from:\n    {url}")


if __name__ == "__main__":
    print("Downloading models...\n")
    for path, url in MODELS.items():
        download(path, url)
    print("\nAll done.")
