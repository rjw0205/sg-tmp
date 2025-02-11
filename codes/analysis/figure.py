from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from cuml import UMAP

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


SCANNERS = [
    "AT2",
    "GT450",
    "DP200",
    "P1000",
    "Philips"
]
REFERENCE_SCANNER = "AT2"
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
COLOR_MAP = {scanner: COLORS[i % len(COLORS)] for i, scanner in enumerate(SCANNERS)}
SYMBOLS = ["o", "s", "D", "v", "^"]
SYMBOL_MAP = {scanner: SYMBOLS[i] for i, scanner in enumerate(SCANNERS)}


def get_image_name(image_path):
    parents = Path(image_path).parents
    image_name = f"{parents[1].name}_{parents[0].name}"
    return image_name


def get_image_paths(image_dir):
    image_paths = defaultdict(list)

    for slide_dir in Path(image_dir).iterdir():
        if not slide_dir.is_dir():
            continue

        for sample_dir in slide_dir.iterdir():
            if not sample_dir.is_dir():
                continue

            for image_path in sample_dir.iterdir():
                if not image_path.is_file() or image_path.suffix != ".jpg":
                    continue

                scanner = image_path.stem.split("_")[0]
                image_paths[scanner].append(str(image_path))

    return image_paths


def get_feature_paths(feature_dir):
    feature_paths = defaultdict(dict)

    for scanner_dir in Path(feature_dir).iterdir():
        if not scanner_dir.is_dir():
            continue

        scanner = scanner_dir.name

        for feature_path in scanner_dir.iterdir():
            if not feature_path.is_file() or feature_path.suffix != ".npy":
                continue

            feature_name = feature_path.stem
            feature_paths[scanner][feature_name] = feature_path

    return feature_paths


def read_image(image_path, color_space):
    img = cv2.imread(image_path)

    if color_space == "HSV":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif color_space == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Color space must be either RGB or HSV")

    return img


def calculate_mean_std(image_paths, color_space):
    means = []
    stds = []

    for image_path in image_paths:
        img = read_image(image_path, color_space)

        channels = cv2.split(img)
        img_means = [np.mean(channel) for channel in channels]
        img_stds = [np.std(channel) for channel in channels]

        means.append(img_means)
        stds.append(img_stds)

    return means, stds


def compute_spatial_domain_statistics(scanner_images):
    color_spaces = ["RGB", "HSV"]
    channel_names = {"RGB": ["R", "G", "B"], "HSV": ["H", "S", "V"]}

    data = []
    for scanner, image_paths in scanner_images.items():
        for color_space in color_spaces:
            means, stds = calculate_mean_std(image_paths, color_space)
            channels = channel_names[color_space]
            
            for i in range(len(image_paths)):
                for j, channel in enumerate(channels):
                    image_data = {
                        "scanner": scanner,
                        "image_name": get_image_name(image_paths[i]),
                        "color_space": color_space,
                        "channel": channel,
                        "mean": means[i][j],
                        "std": stds[i][j],
                    }

                    data.append(image_data)

    df = pd.DataFrame(data)
    return df


def load_features(feature_paths):
    features = []

    for scanner, feature_dict in feature_paths.items():
        for feature_name, feature_path in feature_dict.items():
            feature = {
                "scanner": scanner,
                "feature_name": feature_name,
                "feature": np.load(feature_path)
            }
            features.append(feature)

    df = pd.DataFrame(features)
    return df


def calculate_diff_with_reference(df, bases, targets, reference_scanner):
    new_df = df.copy()

    for _, group in df.groupby(bases):
        for target in targets:
            ref_target = group[group["scanner"] == reference_scanner][target].values
            if ref_target.size == 0:
                raise ValueError(f"Reference not found")

            ref_target = ref_target[0]
            new_df.loc[group.index, target] = group[target].apply(lambda x: x - ref_target)

    return new_df[new_df["scanner"] != reference_scanner]


def reduce_feature(features, target):
    original_features = np.array([feature for feature in features[target]])
    original_features = original_features.reshape(original_features.shape[0], -1)

    reducer = UMAP(random_state=0)
    reduced_features = reducer.fit_transform(original_features)
    for i in range(reduced_features.shape[1]):
        features[f"reduced_{target}_{i}"] = reduced_features[:, i]
    return features


def plot_analysis(statistics, features, path, color_map, reference_scanner):
    rgb_channels = ["R", "G", "B"]

    features = features[features["layer"] == "Layer4"].sort_values(by=["model"])
    models = ["ImageNet"]

    cols = rgb_channels + models
    fig, axes = plt.subplots(1, len(cols), figsize=(12, 2), constrained_layout=True)
    for i, (col, ax) in enumerate(zip(cols, axes)):
        if i < len(rgb_channels):
            col_data = statistics[statistics["channel"] == col]
            x = "mean"
            y = "std"
        else:
            col_data = features[features["model"] == col]
            x = f"reduced_feature_0"
            y = f"reduced_feature_1"
        
        sns.kdeplot(
            data=col_data,
            x=x,
            y=y,
            levels=4,
            fill=False,
            hue="scanner",
            palette=color_map,
            ax=ax,
            legend=False
        )
        
        if reference_scanner not in col_data["scanner"].unique():
            ax.scatter(
                [0],
                [0],
                color=color_map[reference_scanner],
                label=reference_scanner,
                marker="X",
                s=50,
                edgecolor="black",
                zorder=5
            )
        
        ax.set_title(f"{col} channel") if i < len(rgb_channels) else ax.set_title(col)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    
    # Add legend
    custom_legend = [mpatches.Patch(color=color, label=scanner) for scanner, color in color_map.items()]
    fig.legend(handles=custom_legend, loc="upper right", bbox_to_anchor=(1.09, 0.95), title="Scanner")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(image_dir, feature_dirs, output_dir):
    image_paths = get_image_paths(image_dir)

    spatial_domain_stat_path = f"{output_dir}/spatial_domain_statistics.csv"
    spatial_domain_stat_diff_path = f"{output_dir}/spatial_domain_statistics_diff.csv"
    if Path(spatial_domain_stat_path).exists():
        spatial_domain_stat_df = pd.read_csv(spatial_domain_stat_path)
        spatial_domain_stat_diff_df = pd.read_csv(spatial_domain_stat_diff_path)
    else:
        spatial_domain_stat_df = compute_spatial_domain_statistics(image_paths)
        spatial_domain_stat_diff_df = calculate_diff_with_reference(spatial_domain_stat_df, ["image_name", "channel"], ["mean", "std"], REFERENCE_SCANNER)
        spatial_domain_stat_df.to_csv(spatial_domain_stat_path, index=False)
        spatial_domain_stat_diff_df.to_csv(spatial_domain_stat_diff_path, index=False)

    feature_space_stat_path = f"{output_dir}/feature_space_statistics.csv"
    feature_space_stat_diff_path = f"{output_dir}/feature_space_statistics_diff.csv"
    if Path(feature_space_stat_path).exists():
        feature_space_stat_df = pd.read_csv(feature_space_stat_path)
        feature_space_stat_diff_df = pd.read_csv(feature_space_stat_diff_path)
    else:
        feature_space_stat_df = pd.DataFrame()
        feature_space_stat_diff_df = pd.DataFrame()
        for model, per_layer_feature_dir in feature_dirs.items():
            for layer, feature_dir in per_layer_feature_dir.items():
                feature_paths = get_feature_paths(feature_dir)
                df = load_features(feature_paths)
                df["model"] = model
                df["layer"] = layer

                diff_df = calculate_diff_with_reference(df, "feature_name", ["feature"], REFERENCE_SCANNER)
                diff_df = reduce_feature(diff_df, "feature")
                feature_space_stat_diff_df = pd.concat([feature_space_stat_diff_df, diff_df], ignore_index=True)

                reduced_df = reduce_feature(df, "feature")
                feature_space_stat_df = pd.concat([feature_space_stat_df, reduced_df], ignore_index=True)

        feature_space_stat_df.to_csv(feature_space_stat_path, index=False)
        feature_space_stat_diff_df.to_csv(feature_space_stat_diff_path, index=False)

    plot_analysis(spatial_domain_stat_df, feature_space_stat_df, f"{output_dir}/unpaired.pdf", COLOR_MAP, REFERENCE_SCANNER)
    plot_analysis(spatial_domain_stat_diff_df, feature_space_stat_diff_df, f"{output_dir}/paired.pdf", COLOR_MAP, REFERENCE_SCANNER)


if __name__ == "__main__":
    image_dir = "/lunit/data/onco/scope_sg/240409"
    feature_dirs = {
        "ImageNet": {
            "Layer4": "features/imagenet_rn50_layer4_features"
        }
    }
    figure_dir = "figures"

    Path(figure_dir).mkdir(exist_ok=True, parents=True)
    main(image_dir, feature_dirs, figure_dir)
