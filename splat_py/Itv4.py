#4分割*RGB分割の12分割画像への閾値に基づいたフィルタ適用

import torch
from torchvision import transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt 
import os 
import json
import shutil
#ハイパラ管理ファイル
from splat_py.config import SplatConfig

#itv計算関数
def image_total_variation(image_tensor):
    #RGB分割するので二次元テンソルにも対応させる
    if image_tensor.dim() == 2:
        return torch.sum(torch.abs(image_tensor[:-1, :] - image_tensor[1:, :])) + \
            torch.sum(torch.abs(image_tensor[:, :-1] - image_tensor[:, 1:]))
    elif image_tensor.dim() == 3:
        return torch.sum(torch.abs(image_tensor[:, :-1, :] - image_tensor[:, 1:, :])) + \
            torch.sum(torch.abs(image_tensor[:, :, :-1] - image_tensor[:, :, 1:]))
    elif image_tensor.dim() == 4:
        return torch.sum(torch.abs(image_tensor[:, :, :, :-1] - image_tensor[:, :, :, 1:])) + \
            torch.sum(torch.abs(image_tensor[:, :, :-1, :] - image_tensor[:, :, 1:, :]))
            
#変動スコアの1要素平均
def calc_iv_avg(itv,tensor):
    num_elements = tensor.numel()
    iv_avg = itv / num_elements
    return iv_avg

#画像の12分割と計算を行う関数
def divide_calc(image, threshold): #image{RGB}

    #画像はまだtensorではなくnumpy配列
    h,w,c = image.shape 
    mid_h, mid_w =h//2, w//2

    #4分割画像(quadrants)作成
    quadrants = {
        "lt": image[:mid_h, :mid_w],
        "lb": image[mid_h:, :mid_w],
        "rt": image[:mid_h, mid_w:],
        "rb": image[mid_h:, mid_w:]
    }

    filtered_quadrants = {} #フィルタ後の4画像を格納

    for key, quadrant in quadrants.items(): #辞書の要素取得
        #quadrantsをRGBに3分割
        channels = cv2.split(quadrant) 
        filtered_channels = [] #フィルタ後の12画像を格納

        #12分割した画像(channels)をテンソル(channel_tensor)に
        for i, channel in enumerate(channels):
            channel_tensor = torch.tensor(channel, dtype=torch.float32)/255.0

            #itv, iv_avg計算
            itv = image_total_variation(channel_tensor)
            iv_avg = calc_iv_avg(itv, channel_tensor)

            #閾値を超えたものに画像処理
            if iv_avg.item() > threshold:

                #12要素それぞれにフィルタを用いる(channel:iter)
                filtered_channel = cv2.GaussianBlur(channel, (5,5),0) 
                
                #処理後のスコア計算
                p_image_tensor = torch.tensor(filtered_channel, dtype=torch.float32)/255.0
                p_itv = image_total_variation(p_image_tensor)
                p_iv_avg = calc_iv_avg(p_itv, p_image_tensor)

                #処理前後の報告
                print(f"{key} channel {i+1} filtered (IV_AVG:{iv_avg.item()}->{p_iv_avg.item()})") 

            else:
               #閾値を超えていない場合はそのまま出力
                filtered_channel = channel
                print(f"{key} channel {i+1} :フィルタ未適用 (IV_AVG: {iv_avg.item()})")

            #処理済みの12分割画像を格納
            filtered_channels.append(filtered_channel)

        #RGB結合してlt~rbに格納（4分割されたRGB画像へ）
        filtered_quadrants[key] = cv2.merge(filtered_channels)
    
    #元の1画像に戻す(→，↓)
    top_half = cv2.hconcat([filtered_quadrants["lt"], filtered_quadrants["rt"]]) 
    bottom_half = cv2.hconcat([filtered_quadrants["lb"], filtered_quadrants["rb"]])
    filtered_image = cv2.vconcat([top_half, bottom_half])

    return filtered_image

def main():
    dataset_path = "D:\Data\itv\dataset1"

    #出力フォルダを作成
    output_dir = "D:\Data\itv\processed_dataset2"
    os.makedirs(output_dir, exist_ok=True)

    #指定ディレクトリ内の各ファイル(name)を取得
    for name in os.listdir(dataset_path):

        #各画像へのパスを作成(path.join(dir,name))-> dir\name
        i_path = os.path.join(dataset_path,name)

        #各画像を読み込む
        imageBGR = cv2.imread(i_path) 
        image = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)

        #RGB画像(numpy)で計算
        print(f"{os.path.basename(name)}")
        filtered_image = divide_calc(image,Config.IV_AVG_THRESHOLD)
 
        #出力ディレクトリに保存
        output_path = os.path.join(output_dir, name) #input\name -> outdir\name
        filtered_image_BGR = cv2.cvtColor(filtered_image, cv2.COLOR_RGB2BGR) #cv.imwriteはBGRが前提
        cv2.imwrite(output_path, filtered_image_BGR) #パスに書き込み

#mainの時のみ実行
if __name__=="__main__":
    main()