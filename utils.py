import cv2
import torch
import shapely.wkt as wkt
from skimage import draw


def read_img(img_path):
    """Load image based on cv2 library.
    
    Parameters
    ----------
    img_path: str
        path of image

    Returns
    -------
    img: np.ndarray
        numpy RGB order image
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def parse_wkt_annotation(wkt_path):
    """Load wkt label and parse it into list of points.
    
    Parameters
    ----------
    wkt_path: str
        path of wkt label

    Returns
    -------
    gt_points: List[Tuple(int, int)]
        A list of points (x, y).
    gt_categories: List[int]
        A list of category.
    """
    gt_points = []
    gt_categories = []
    with open(wkt_path, "r") as f:
        for line in f.read().splitlines():
            point, category = line.split("|")
            point = wkt.loads(point)
            gt_points.append((int(point.x), int(point.y)))
            gt_categories.append(int(category))

    return gt_points, gt_categories

def draw_segmentation_label(img, gt_points, gt_categories, radius):
    """Convert a list of points to a segmentation map containing circles.

    Parameters
    ----------
    img: torch.tensor
        An input image of size 3 x H x W.
    gt_points: List[Tuple(int, int)]
        A list of points (x, y).
    gt_categories: List[int]
        A list of category.
    radius: int
        A radius used for drawing circles.

    Returns
    -------
    seg_label: torch.tensor
        A tensor of size H x W that contains zeros by default, and circles otherwise. 
        Each circle is filled with a value of class index.
    """
    H, W = img.shape[1:]
    seg_label = torch.zeros((H, W), dtype=torch.int64)
    for (x, y), category in zip(gt_points, gt_categories):
        circle_y, circle_x = draw.disk((int(y), int(x)), radius, shape=(H, W))
        seg_label[circle_y, circle_x] = int(category)

    return seg_label
