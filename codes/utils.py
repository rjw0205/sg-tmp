import cv2
import torch
import numpy as np
import shapely.wkt as wkt
from skimage import draw
from PIL import Image
from skimage.feature import peak_local_max


def save_img_from_numpy_array(arr, save_name):
    """ Save image from numpy array (HxWxC)

    Parameters
    ----------
    arr: np.ndarray
        An array to be saved as an image.
    save_name: str
        Image name to be saved.
    """
    img = Image.fromarray(arr)
    img.save(save_name)


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


def find_peaks_from_heatmap(arr, min_distance):
    """ Find peaks from heatmap.

    Parameters
    ----------
    arr: torch.tensor
        An array shape (1 + num_classes, H, W), where 1 denotes Background.

    min_distance: int
        Minimum distance (pixel) between peaks.

    Returns
    ------------
    peaks_coords: np.ndarray (N, 2)
        Coordinates of peaks.
    
    peaks_score: np.ndarray (N,)
        Score of peaks.

    peaks_class: np.ndarray (N,)
        Class of peaks
    """
    arr = np.array(arr)
    assert np.all(np.isclose(np.sum(arr, axis=0), 1.0)), "Input arr should be post-softmax."

    # Use background channel for peak finding
    bkg = arr[0, :, :]
    obj = 1.0 - bkg

    # Coords are (y, x) order
    peaks_coords = peak_local_max(obj, min_distance=min_distance)
    peaks_score = np.max(arr, axis=0)[peaks_coords[:, 0], peaks_coords[:, 1]]
    peaks_class = np.argmax(arr, axis=0)[peaks_coords[:, 0], peaks_coords[:, 1]]

    # Filter out only mitotic cells (class index 1)
    is_mitotic_cell = (peaks_class == 1)
    peaks_coords = peaks_coords[is_mitotic_cell]
    peaks_score = peaks_score[is_mitotic_cell]
    peaks_class = peaks_class[is_mitotic_cell]

    if len(peaks_coords) == 0:
        return np.empty((0, 2)), np.empty((0)), np.empty((0))

    return peaks_coords, peaks_score, peaks_class
