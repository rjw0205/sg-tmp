from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
from cuml import UMAP
from plotly.subplots import make_subplots


SCANNERS = [
    "AT2",
    "GT450",
    "DP200",
    "P1000",
    "Philips"
]
REFERENCE_SCANNER = "AT2"
COLORS = pc.qualitative.Plotly
COLOR_MAP = {scanner: COLORS[i % len(COLORS)] for i, scanner in enumerate(SCANNERS)}


def get_image_name(image_path):
    parents = Path(image_path).parents
    image_name = f"{parents[1].name}_{parents[0].name}"
    return image_name


def get_image_paths(data_dir):
    image_paths = defaultdict(list)

    for slide_dir in Path(data_dir).iterdir():
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


def calculate_amplitude_power_circular_variance(image_paths, color_space, amplitude_power_percentages):
    amplitude_powers = []
    circular_variances = []
    for image_path in image_paths:
        img = read_image(image_path, color_space)
        
        channels = cv2.split(img)
        amplitude_power_per_channel = []
        circular_variance_per_channel = []
        for channel in channels:
            f = np.fft.fft2(channel)
            fshift = np.fft.fftshift(f)  # Center the low frequencies
            amplitude = np.abs(fshift)
            phase = np.angle(fshift)

            power_spectrum = amplitude ** 2
            rows, cols = power_spectrum.shape
            crow, ccol = rows // 2 , cols // 2  # Center of the frequency spectrum
            y, x = np.ogrid[:rows, :cols]

            # Create a circular mask for low frequencies
            low_freq_power_at_percentage = {}
            for amplitude_power_percentage in amplitude_power_percentages:
                radius = int(min(rows, cols) * amplitude_power_percentage / 100)  # Radius for low-frequency region
                mask = (x - ccol)**2 + (y - crow)**2 <= radius**2  # Circular mask

                # Apply the mask to the power spectrum and sum to get low-frequency power
                low_freq_power = np.sum(power_spectrum * mask)
                low_freq_power_at_percentage[amplitude_power_percentage] = low_freq_power
            amplitude_power_per_channel.append(low_freq_power_at_percentage)

            # Calculate circular variance
            N = phase.size
            circular_variance = 1 - np.abs(np.sum(np.exp(1j * phase))) / N
            circular_variance_per_channel.append(circular_variance)

        amplitude_powers.append(amplitude_power_per_channel)
        circular_variances.append(circular_variance_per_channel)

    return amplitude_powers, circular_variances


def compute_statistics(scanner_images, amplitude_power_percentages):
    color_spaces = ['RGB', 'HSV']
    channel_names = {'RGB': ['R', 'G', 'B'], 'HSV': ['H', 'S', 'V']}

    data = []
    for scanner, image_paths in scanner_images.items():
        for color_space in color_spaces:
            means, stds = calculate_mean_std(image_paths, color_space)
            amplitude_powers, circular_variances = calculate_amplitude_power_circular_variance(image_paths, color_space, amplitude_power_percentages)
            channels = channel_names[color_space]
            
            for i in range(len(image_paths)):
                for j, channel in enumerate(channels):
                    image_data = {
                        'scanner': scanner,
                        'image_index': get_image_name(image_paths[i]),
                        'color_space': color_space,
                        'channel': channel,
                        'mean': means[i][j],
                        'std': stds[i][j],
                        'circular_variance': circular_variances[i][j]
                    }
                    for percentage, amplitude_power in amplitude_powers[i][j].items():
                        image_data[f"amplitude_power_at_low_{percentage}p"] = amplitude_power

                    data.append(image_data)

    df = pd.DataFrame(data)
    return df


def load_features(feature_paths):
    features = []
    for scanner, feature_dict in feature_paths.items():
        for feature_name, feature_path in feature_dict.items():
            feature = {
                'scanner': scanner,
                'feature_name': feature_name,
                'feature_path': feature_path,
                'feature': np.load(feature_path)
            }
            features.append(feature)

    df = pd.DataFrame(features)
    return df


def calculate_subtraction(df, base, target, reference_scanner):
    df[f'{target}_diff'] = None

    for _, group in df.groupby(base):
        ref_target = group[group['scanner'] == reference_scanner][target].values
        if ref_target.size == 0:
            raise ValueError(f"Reference not found")

        ref_target = ref_target[0]
        df.loc[group.index, f'{target}_diff'] = group[target].apply(lambda x: x - ref_target)
    
    return df


def reduce_feature(features, target, n_neighbors):
    original_features = np.array([feature for feature in features[target]])
    original_features = original_features.reshape(original_features.shape[0], -1)

    reducer = UMAP(n_neighbors=n_neighbors, random_state=0)
    reduced_features = reducer.fit_transform(original_features)
    for i in range(reduced_features.shape[1]):
        features[f"reduced_{target}_{i}"] = reduced_features[:, i]
    return features


def show_line(df, target, path, reference_scanner):
    channels = df['channel'].unique()
    fig = make_subplots(rows=1, cols=len(channels), subplot_titles=[f"{channel} Channel" for channel in channels])

    unique_scanners = df['scanner'].unique()
    for col, channel in enumerate(channels, start=1):
        reference_df = df[(df['channel'] == channel) & (df['scanner'] == reference_scanner)]
        sorted_indices = reference_df.sort_values(by=target)['image_index']

        for scanner in unique_scanners:
            scanner_df = df[(df['channel'] == channel) & (df['scanner'] == scanner)]
            
            aligned_values = scanner_df.set_index('image_index').reindex(sorted_indices)[target]
            
            fig.add_trace(
                go.Scatter(
                    x=sorted_indices,
                    y=aligned_values,
                    mode='lines+markers',
                    name=scanner,
                    showlegend=col == 1,
                    legendgroup=scanner,
                    line=dict(color=COLOR_MAP[scanner])
                ),
                row=1, col=col
            )

    fig.update_layout(
        title=f"{target.capitalize()} Comparison Across Scanners (Ordered by {reference_scanner} {target.capitalize()})",
        xaxis_title='Ordered Image Index',
        yaxis_title=target.capitalize(),
        legend_title="Scanners"
    )

    for i in range(1, len(channels) + 1):
        fig['layout'][f'xaxis{i}']['title'] = "Image Index"
        fig['layout'][f'xaxis{i}']['showticklabels'] = False

    fig.write_html(path)


def show_mean_std(df, path, showdiff):
    fig = px.scatter(
        df,
        x="mean_diff" if showdiff else "mean",
        y="std_diff" if showdiff else "std",
        color="scanner",
        color_discrete_map=COLOR_MAP,
        facet_col="channel",
        symbol="scanner",
        title="Mean vs. Standard Deviation for RGB and HSV Channels Across Scanners",
        labels={'mean': 'Mean', 'std': 'Standard Deviation'},
    )

    fig.update_layout(legend_title_text='Channel')
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.update_yaxes(matches=None, showticklabels=True)

    fig.write_html(path)


def show_amplitude_power_circle_variance(df, path, percentages, showdiff):
    for percentage in percentages:
        fig = px.scatter(
            df,
            x=f"amplitude_power_at_low_{percentage}p_diff" if showdiff else f"amplitude_power_at_low_{percentage}p",
            y="circular_variance_diff" if showdiff else "circular_variance",
            color="scanner",
            color_discrete_map=COLOR_MAP,
            facet_col="channel",
            symbol="scanner",
            title=f"Amplitude Power (Low {percentage}%) vs. Circular Variance for RGB and HSV Channels Across Scanners",
            labels={f"low_{percentage}p_profile": f'Amplitude Power (Low {percentage}%)', 'circular_variance': 'Circular Variance'},
        )

        fig.update_layout(legend_title_text='Channel')
        fig.update_xaxes(matches=None, showticklabels=True)
        fig.update_yaxes(matches=None, showticklabels=True)

        file_path = Path(path).with_stem(f"low_{percentage}p_" + Path(path).stem)
        fig.write_html(file_path)


def show_features(features, target, path):
    fig = px.scatter(
        features,
        x=f"reduced_{target}_0",
        y=f"reduced_{target}_1",
        color='scanner',
        color_discrete_map=COLOR_MAP,
        symbol='scanner',
        title="UMAP Visualization of Features",
        labels={'x': 'UMAP 1', 'y': 'UMAP 2'},
    )

    fig.update_layout(legend_title_text='Scanner')
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.update_yaxes(matches=None, showticklabels=True)

    fig.write_html(path)


def main(data_dir, feature_dirs, output_dir):
    amplitude_power_percentages = [1, 5, 10, 50]

    image_paths = get_image_paths(data_dir)
    stat_path = f"{output_dir}/statistics.csv"
    if Path(stat_path).exists():
        df = pd.read_csv(stat_path)
    else:
        df = compute_statistics(image_paths, amplitude_power_percentages)
        df.to_csv(stat_path, index=False)

    df = calculate_subtraction(df, ["image_index", "channel"], 'mean', REFERENCE_SCANNER)
    df = calculate_subtraction(df, ["image_index", "channel"], 'std', REFERENCE_SCANNER)
    show_mean_std(df, f"{output_dir}/mean_std.html", showdiff=False)
    show_mean_std(df[df['scanner'] != REFERENCE_SCANNER], f"{output_dir}/mean_std_diff.html", showdiff=True)

    for percentage in amplitude_power_percentages:
        df = calculate_subtraction(df, ["image_index", "channel"], f"amplitude_power_at_low_{percentage}p", REFERENCE_SCANNER)
    df = calculate_subtraction(df, ["image_index", "channel"], 'circular_variance', REFERENCE_SCANNER)
    show_amplitude_power_circle_variance(df, f"{output_dir}/radial_profile_circular_variance.html", amplitude_power_percentages, showdiff=False)
    show_amplitude_power_circle_variance(df[df['scanner'] != REFERENCE_SCANNER], f"{output_dir}/radial_profile_circular_variance_diff.html", amplitude_power_percentages, showdiff=True)

    show_line(df, "mean", f"{output_dir}/mean.html", REFERENCE_SCANNER)
    show_line(df, "std", f"{output_dir}/std.html", REFERENCE_SCANNER)
    show_line(df, "circular_variance", f"{output_dir}/circular_variance.html", REFERENCE_SCANNER)

    for percentage in amplitude_power_percentages:
        show_line(df, f"amplitude_power_at_low_{percentage}p", f"{output_dir}/amplitude_power_{percentage}p.html", REFERENCE_SCANNER)

    n_neighbors = 10
    for feature_dir in feature_dirs:
        feature_paths = get_feature_paths(feature_dir)
        features = load_features(feature_paths)

        features = reduce_feature(features, 'feature', n_neighbors)
        show_features(features, 'feature', f"{output_dir}/{feature_dir}_{n_neighbors}.html")

        features = calculate_subtraction(features, 'feature_name', 'feature', REFERENCE_SCANNER)
        features = reduce_feature(features, 'feature_diff', n_neighbors)
        show_features(features[features['scanner'] != REFERENCE_SCANNER], 'feature_diff', f"{output_dir}/{feature_dir}_diff_{n_neighbors}.html")


if __name__ == "__main__":
    data_dir = "/lunit/data/onco/scope_sg/240409"
    feature_dirs = ["bt_rn50_ep200_layer1_features", "bt_rn50_ep200_layer4_features", "imagenet_rn50_layer1_features", "imagenet_rn50_layer4_features"]
    output_dir = "outputs"

    Path(output_dir).mkdir(exist_ok=True, parents=True)
    main(data_dir, feature_dirs, output_dir)
