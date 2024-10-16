import cv2
import torch
import numpy as np
import shapely.wkt as wkt
from skimage import draw
from PIL import Image
from skimage.feature import peak_local_max
from codes.constant import MITOTIC_CELL_CLS_IDX
from torchvision import transforms
from PIL import Image, ImageDraw


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
    """Load wkt label and parse it into list of points. Only mitotic cells are treated as GT.
    
    Parameters
    ----------
    wkt_path: str
        path of wkt label

    Returns
    -------
    gt_coords: List[Tuple(int, int)]
        A list of points (y, x).
    gt_categories: List[int]
        A list of category.
    """
    gt_coords = []
    gt_categories = []
    with open(wkt_path, "r") as f:
        for line in f.read().splitlines():
            point, category = line.split("|")
            if int(category) == MITOTIC_CELL_CLS_IDX:  # ignore non mitotic cells
                point = wkt.loads(point)
                gt_coords.append((int(point.y), int(point.x)))
                gt_categories.append(int(category))

    return gt_coords, gt_categories

def draw_segmentation_label(img, gt_coords, gt_categories, radius):
    """Convert a list of points to a segmentation map containing circles.

    Parameters
    ----------
    img: torch.tensor
        An input image of size 3 x H x W.
    gt_coords: List[Tuple(int, int)]
        A list of points (y, x).
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
    for (y, x), category in zip(gt_coords, gt_categories):
        circle_y, circle_x = draw.disk((int(y), int(x)), radius, shape=(H, W))
        seg_label[circle_y, circle_x] = int(category)

    return seg_label


def find_mitotic_cells_from_heatmap(arr, min_distance, max_num_peaks=10):
    """ Find mitotic cells from prediction heatmap.

    Parameters
    ----------
    arr: np.ndarray
        An array shape (1 + num_classes, H, W), where 1 denotes background.

    mitotic_cell_cls_idx: int
        A class index of mitotic cell.

    min_distance: int
        A minimum distance (pixel) between cells.

    max_num_peaks: int
        A maximum number of peaks that can exist on an image.

    Returns
    ------------
    cell_coords: np.ndarray (N, 2)
        Coordinates of cells.
    
    cell_scores: np.ndarray (N,)
        Confidence probility of cells.
    """
    assert isinstance(arr, np.ndarray)
    assert np.all(np.isclose(np.sum(arr, axis=0), 1.0)), "Input arr should be post-softmax."
    assert arr.shape[0] == 2, "Currently only support num_classes 2."

    obj = arr[1, :, :]
    cell_coords = peak_local_max(
        obj, 
        min_distance=min_distance, 
        threshold_abs=0.5, 
        exclude_border=False, 
        num_peaks=max_num_peaks,
    )
    cell_scores = obj[cell_coords[:, 0], cell_coords[:, 1]]

    if len(cell_coords) == 0:
        return np.empty((0, 2)), np.empty((0))

    return cell_coords, cell_scores


def compute_tp_and_fp(pred_coords, pred_scores, gt_coords, distance_cut_off):
    """ Compute TP and FP from given prediction and GT coordinates.

    Parameters
    ----------
    pred_coords: np.ndarray (N, 2)
        List of predicted cell coordinates. N is number of predictions, 2 corresponds to (y, x).

    pred_scores: np.ndarray (N,)
        List of cell's confidence score.

    gt_coords: np.ndarray (M, 2)
        List of GT cell coordinates. M is number of GT, 2 corresponds to (y, x).

    distance_cut_off: int
        Distance (pixel) threshold which will decided the match between pred and GT cells.

    Returns
    -------
    num_tp: int
        Number of True Positive (TP) detection.

    num_fp: int
        Number of False Positive (FP) detection.
    """
    num_tp, num_fp = 0, 0
    num_preds = pred_coords.shape[0]

    # Compute distance between GT and predicted cells
    pred_coords = pred_coords.reshape([-1, 1, 2])
    gt_coords = gt_coords.reshape([1, -1, 2])
    distance = np.linalg.norm(pred_coords - gt_coords, axis=2)

    # Start matching from highest confidence predicted cell
    sorted_pred_indices = np.argsort(-pred_scores)
    bool_mask = (distance <= distance_cut_off)
    for pred_idx in sorted_pred_indices:
        gt_neighbors = bool_mask[pred_idx].nonzero()[0]
        if len(gt_neighbors) == 0:  # No matching GT --> False Positive
            num_fp += 1
        else: # Assign nearest GT --> True Positive
            gt_idx = min(gt_neighbors, key=lambda gt_idx: distance[pred_idx, gt_idx])
            num_tp += 1
            bool_mask[:, gt_idx] = False

    assert num_tp + num_fp == num_preds
    return num_tp, num_fp


def compute_precision_recall_f1(num_gt, num_tp, num_fp, eps=1e-7):
    """

    Parameters
    ----------
    num_gt: int
        Number of GT cells.

    num_tp: int
        Number of True Positive (TP) detection.

    num_fp: int
        Number of False Positive (FP) detection.

    eps: float
        A very small number to prevent ZeroDivisionError.

    Returns
    -------
    precision: float
    recall: float
    f1: float
    """
    precision = num_tp / (num_tp + num_fp + eps)
    recall = num_tp / (num_gt + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


def save_visualization(img, pred, pred_coords, gt_coords, save_path):
    """
    img: torch.tensor
        Image tensor with shape (3, H, W) with RGB channel order.
    
    pred: np.ndarray
        Model output after softmax (num_class, H, W).

    pred_coords: np.ndarray
        List of (y, x) coordinates which are predictions.

    gt_coords: np.ndarray
        List of (y, x) coordinates which are GT.

    save_path: str
        Path to save a visualization.
    """
    radius = 10
    margin = 30

    # Create original image
    org_img = transforms.ToPILImage()(img)

    # Create prediction overlayed image
    pred_img = org_img.copy()
    pred_draw = ImageDraw.Draw(pred_img)
    for pred_coord in pred_coords:
        pred_y, pred_x = pred_coord
        left_top = (pred_x - radius, pred_y - radius)
        right_bot = (pred_x + radius, pred_y + radius)
        pred_draw.ellipse([left_top, right_bot], outline="red", width=3)

    # Object heatmap
    assert pred.shape[0] == 2
    pred_heatmap = (pred[1, :, :] * 255).astype(np.uint8)
    pred_heatmap = np.stack([pred_heatmap] * 3, axis=-1)
    pred_heatmap_img = Image.fromarray(pred_heatmap)

    # Create prediction overlayed image
    gt_img = org_img.copy()
    gt_draw = ImageDraw.Draw(gt_img)
    for gt_coord in gt_coords:
        gt_y, gt_x = gt_coord
        left_top = (gt_x - radius, gt_y - radius)
        right_bot = (gt_x + radius, gt_y + radius)
        gt_draw.ellipse([left_top, right_bot], outline="red", width=3)

    # Concatenate and save a final image
    width, height = org_img.size
    final_image = Image.new("RGB", (width * 4 + margin * 3, height), (255, 255, 255))
    final_image.paste(org_img, (0, 0))
    final_image.paste(pred_heatmap_img, (width + margin, 0))
    final_image.paste(pred_img, (width * 2 + margin * 2, 0))
    final_image.paste(gt_img, (width * 3 + margin * 3, 0))
    final_image.save(save_path)