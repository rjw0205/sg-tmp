import cv2
import numpy as np


def fda_augmentation(src_img, tgt_img, channel, L):
    assert channel in ("H", "S", "V")
    assert 0.0 <= L <= 0.5

    # Convert images from RGB to HSV
    src_img_hsv = cv2.cvtColor(src_img, cv2.COLOR_RGB2HSV)
    tgt_img_hsv = cv2.cvtColor(tgt_img, cv2.COLOR_RGB2HSV)

    # Select channel to perform FDA
    channel_idx = ["H", "S", "V"].index(channel)
    src_channel_to_fda = src_img_hsv[:, :, channel_idx]
    tgt_channel_to_fda = tgt_img_hsv[:, :, channel_idx]

    # Perform FDA
    src_img_hsv[:, :, channel_idx] = swap_amplitude(src_channel_to_fda, tgt_channel_to_fda, L)

    # Reconstruct image after FDA
    fda_img = cv2.cvtColor(src_img_hsv, cv2.COLOR_HSV2RGB)
    return fda_img


def swap_amplitude(src_channel, tgt_channel, L):
    assert src_channel.ndim == 2 and tgt_channel.ndim == 2

    original_src_channel = src_channel.copy()

    # get fft of both source and target
    fft_src = np.fft.fft2(src_channel)
    fft_tgt = np.fft.fft2(tgt_channel)

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src), np.angle(fft_src)
    amp_tgt, pha_tgt = np.abs(fft_tgt), np.angle(fft_tgt)

    # mutate the amplitude part of source with target
    s = amp_src.shape[0]
    start, end = int(s * (0.5 - L)), int(s * (0.5 + L))
    amp_src = np.fft.fftshift(amp_src)
    amp_tgt = np.fft.fftshift(amp_tgt)
    amp_src[start:end, start:end] = amp_tgt[start:end, start:end]
    amp_src = np.fft.ifftshift(amp_src)

    # mutated fft of source
    fft_src = amp_src * np.exp(1j * pha_src)

    # get the mutated image
    src_channel = np.fft.ifft2(fft_src)
    src_channel = np.clip(np.real(src_channel), 0, 255).astype(np.uint8)

    return src_channel