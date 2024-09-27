import cv2
import shapely.wkt as wkt


def read_img(img_path):
    """Load image based on cv2 library. The return image should be 'RGB' order image."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def read_wkt(wkt_path):
    with open(wkt_path, "r") as f:
        wkt_label = f.read().splitlines()
        return wkt_label