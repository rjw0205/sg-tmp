import os
import glob
import multiprocessing as mp
import openslide
import json
import pickle
from tqdm import tqdm
from pathlib import Path
from shapely.geometry import Point
from shapely.wkt import dumps


ROOT_PATH = "/storage6/pp/share/rjw0205/scanner_generalization/paper_submission/MIDOG_2021"
SAVE_PATH = "midog_2021_patches"
PATCH_SIZE = 512
STEP_SIZE = 256


def _parse_annotation(json_path):
    with open(json_path, "r") as f:
        d = json.load(f)
        annotations = d["annotations"]
        images = d["images"]

    result = {}
    for image in images:
        file_name = image["file_name"]
        image_id = image["id"]
        result[file_name] = [annot for annot in annotations if annot["image_id"] == image_id]

    return result


def _sld_idx_to_scanner(sld_idx):
    idx = int(sld_idx)
    if idx <= 50:
        scanner = "Hamamatsu_XR"
    elif idx <= 100:
        scanner = "Hamamatsu_S360"
    elif idx <= 150:
        scanner = "Aperio_CS2"
    elif idx <= 200:
        scanner = "Leica_GT450"
    else:
        raise ValueError(f"Slide index should be 001 ~ 200, but got {sld_idx}.")

    return scanner


def extract_patches_and_labels_from_slide(sld_path, annotation):
    sld_idx = Path(sld_path).stem
    scanner = _sld_idx_to_scanner(sld_idx)
    os.makedirs(f"{SAVE_PATH}/{scanner}/{sld_idx}", exist_ok=True)

    sld = openslide.open_slide(sld_path)
    width, height = sld.dimensions

    metadata = {}
    for start_x in range(0, width - STEP_SIZE + 1, STEP_SIZE):
        for start_y in range(0, height - STEP_SIZE + 1, STEP_SIZE):
            # Save patch
            save_name = f"{SAVE_PATH}/{scanner}/{sld_idx}/x_{start_x}_y_{start_y}_size_{PATCH_SIZE}"
            roi = sld.read_region((start_x, start_y), 0, (PATCH_SIZE, PATCH_SIZE))
            roi = roi.convert("RGB")
            roi.save(f"{save_name}.jpg", "JPEG")

            # Prepare metadata contains cell count information
            metadata[save_name] = {
                "slide_id": sld_idx,
                "patch_id": f"x_{start_x}_y_{start_y}_size_{PATCH_SIZE}",
                "scanner": scanner,
                "num_mitotic_figure": 0, 
                "num_non_mitotic_figure": 0,
            }

            # Save label as wkt file
            with open(f"{save_name}.wkt", "w") as f:
                for cell in annotation:
                    cell_x_coord = int((cell["bbox"][0] + cell["bbox"][2]) // 2)
                    cell_y_coord = int((cell["bbox"][1] + cell["bbox"][3]) // 2)
                    cell_label = cell["category_id"]
                    if (start_x <= cell_x_coord < start_x + PATCH_SIZE) and (start_y <= cell_y_coord < start_y + PATCH_SIZE):
                        f.write(f"POINT ({cell_x_coord} {cell_y_coord})|{cell_label}" + "\n")
                        if cell_label == 1:
                            metadata[save_name]["num_mitotic_figure"] += 1
                        elif cell_label == 2:
                            metadata[save_name]["num_non_mitotic_figure"] += 1
                        else:
                            raise ValueError(f"Cell label should be either 1 or 2, but {cell_label}.")

    print(f"Done {Path(sld_path).name}")
    return metadata


if __name__ == "__main__":
    # Prepare slide paths
    sld_paths = sorted(glob.glob(f"{ROOT_PATH}/*.tiff"))
    assert len(sld_paths) == 200 and isinstance(sld_paths, list)

    # Prepare annotations
    annotations = _parse_annotation(f"{ROOT_PATH}/MIDOG.json")
    assert len(annotations) == 200 and isinstance(annotations, dict)

    # Extract images, labels and metadata
    args = [(sld_path, annotations[Path(sld_path).name]) for sld_path in sld_paths]
    with mp.Pool(processes=mp.cpu_count() // 2) as pool:
        results = pool.starmap(extract_patches_and_labels_from_slide, args)

    # Save metadata
    merged_results = {k: v for result in results for k, v in result.items()}
    with open(f"{SAVE_PATH}/metadata.pkl", "wb") as f:
        pickle.dump(merged_results, f)
