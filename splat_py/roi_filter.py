# ROI-Filtered Images作成(n^2*RGB(3n^2)分割)
#--------------------------------------------------
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt 
import os 
import statistics
import math

#--------------------------------------------------
# itv計算関数
def image_total_variation(image_tensor):
    if image_tensor.dim() == 2:
        return torch.sum(torch.abs(image_tensor[:-1, :] - image_tensor[1:, :])) + \
            torch.sum(torch.abs(image_tensor[:, :-1] - image_tensor[:, 1:]))
    elif image_tensor.dim() == 3:
        return torch.sum(torch.abs(image_tensor[:, :-1, :] - image_tensor[:, 1:, :])) + \
            torch.sum(torch.abs(image_tensor[:, :, :-1] - image_tensor[:, :, 1:]))
    elif image_tensor.dim() == 4:
        return torch.sum(torch.abs(image_tensor[:, :, :, :-1] - image_tensor[:, :, :, 1:])) + \
            torch.sum(torch.abs(image_tensor[:, :, :-1, :] - image_tensor[:, :, 1:, :]))
            
# iv_avg（itvの1要素平均）計算
def calc_iv_avg(itv, tensor):
    # テンソルの次元を取得
    dims = tensor.dim()
    if dims == 2:  # 2次元（高さ×幅のグレースケール画像など）
        h, w = tensor.shape
        num_variations = h * (w - 1) + (h - 1) * w
    elif dims == 3:  # 3次元（RGB画像など、チャネル×高さ×幅）
        c, h, w = tensor.shape
        num_variations = c * (h * (w - 1) + (h - 1) * w)
    elif dims == 4:  # 4次元（バッチサイズ×チャネル×高さ×幅）
        b, c, h, w = tensor.shape
        num_variations = b * c * (h * (w - 1) + (h - 1) * w)
    else:
        raise ValueError("Unsupported tensor dimension. Expected 2D, 3D, or 4D tensor.")

    iv_avg = itv / num_variations
    return iv_avg

# 閾値(τTV)判定＋ROIにフィルタ処理
def divide_calc(image, threshold, divisions):
    h, w, c = image.shape
    step_h = math.ceil(h / divisions)
    step_w = math.ceil(w / divisions)

    mask = np.zeros((h, w), dtype=np.uint8)  # 閾値を超えた領域を記録するマスク
    iv_avg_list = []
    filter_count = 0
    total_elements = divisions * divisions * 3  # RGB 各チャンネルの総ブロック数

    for i in range(divisions):
        for j in range(divisions):
            start_h, end_h = i * step_h, min((i + 1) * step_h, h)
            start_w, end_w = j * step_w, min((j + 1) * step_w, w)

            segment = image[start_h:end_h, start_w:end_w]
            channels = cv2.split(segment)

            # 各チャネルのiv_avg計算
            for channel in channels:
                channel_tensor = torch.tensor(channel, dtype=torch.float32) / 255.0
                itv = image_total_variation(channel_tensor)
                iv_avg = calc_iv_avg(itv, channel_tensor)
                iv_avg_list.append(iv_avg.item())

                # 閾値(τTV)を超えたらマスクを更新
                if iv_avg.item() > threshold:
                    mask[start_h:end_h, start_w:end_w] = 1
                    filter_count += 1  # フィルタ適用回数カウント

    # ROIのみガウシアンブラー処理
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    filtered_image = np.where(mask[:, :, np.newaxis] == 1, blurred_image, image)

    return filtered_image, iv_avg_list, filter_count, total_elements


def calculate_cdf(iv_avg_list):
    """
    IV_AVGの累積分布関数（CDF）を計算し、統計的情報を返す。
    
    Args:
        iv_avg_list (list of float): IV_AVGの値のリスト。
    
    Returns:
        dict: CDFと統計的情報を含む辞書。
            - "cdf" (numpy.ndarray): 累積分布関数。
            - "sorted_values" (numpy.ndarray): ソート済みのIV_AVG値。
            - "mean" (float): 平均値。
            - "std_dev" (float): 標準偏差。
            - "thresholds" (dict): 上位割合の閾値情報。
    """
    sorted_values = np.sort(iv_avg_list)  # ソートされたIV_AVG値
    cdf = np.cumsum(np.ones_like(sorted_values)) / len(sorted_values)  # 累積分布

    # 統計的指標
    mean = np.mean(iv_avg_list)
    std_dev = np.std(iv_avg_list)

    # 上位割合の閾値
    thresholds = {
        "top_5_percent": sorted_values[int(0.95 * len(sorted_values))],
        "top_10_percent": sorted_values[int(0.90 * len(sorted_values))],
        "bottom_5_percent": sorted_values[int(0.05 * len(sorted_values))],
        "bottom_10_percent": sorted_values[int(0.10 * len(sorted_values))]
    }

    return {
        "cdf": cdf,
        "sorted_values": sorted_values,
        "mean": mean,
        "std_dev": std_dev,
        "thresholds": thresholds,
    }
