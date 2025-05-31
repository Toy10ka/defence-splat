import os
import cv2
import torch

from splat_py.config import SplatConfig
from splat_py.read_colmap import (
    read_images_binary,
    read_points3D_binary,
    read_cameras_binary,
    qvec2rotmat,
)
from splat_py.utils import inverse_sigmoid, compute_initial_scale_from_sparse_points
from splat_py.structs import Gaussians, Image, Camera
#追加
from splat_py.roi_filter import divide_calc, calculate_cdf 
import statistics
from pathlib import Path

class GaussianSplattingDataset:
    """
    Generic Gaussian Splatting Dataset class

    Classes that inherit from this class should have the following variables:

    device: torch device
    xyz: Nx3 tensor of points
    rgb: Nx3 tensor of rgb values

    images: list of Image objects
    cameras: dict of Camera objects

    """

    def __init__(self, config):
        self.config = config

    def verify_loaded_points(self):
        """
        Verify that the values loaded from the dataset are consistent
        """
        N = self.xyz.shape[0]
        assert self.xyz.shape[1] == 3
        assert self.rgb.shape[0] == N
        assert self.rgb.shape[1] == 3

    def create_gaussians(self):
        """
        Create gaussians object from the dataset
        """
        self.verify_loaded_points()

        N = self.xyz.shape[0]
        initial_opacity = torch.ones(N, 1) * inverse_sigmoid(self.config.initial_opacity)
        # compute scale based on the density of the points around each point
        initial_scale = compute_initial_scale_from_sparse_points(
            self.xyz,
            num_neighbors=self.config.initial_scale_num_neighbors,
            neighbor_dist_to_scale_factor=self.config.initial_scale_factor,
            max_initial_scale=self.config.max_initial_scale,
        )
        initial_quaternion = torch.zeros(N, 4)
        initial_quaternion[:, 0] = 1.0

        return Gaussians(
            xyz=self.xyz.to(self.device),
            rgb=self.rgb.to(self.device),
            opacity=initial_opacity.to(self.device),
            scale=initial_scale.to(self.device),
            quaternion=initial_quaternion.to(self.device),
        )

    def get_images(self):
        """
        get images from the dataset
        """
        return self.images
#images=colmap_data.get_images()，trainer = SplatTrainer(~,images~)なのでoriginも追加
    def get_images_origin(self):

        return self.images_origin

#add
    def get_images_clean(self):

        return self.images_clean
    
    def get_cameras(self):
        """
        get cameras from the dataset
        """

        return self.cameras


class ColmapData(GaussianSplattingDataset):
    """
    This class loads data similar to Mip-Nerf 360 Dataset generated with colmap

    Format:

    dataset_dir:
        images: full resoloution images
            ...
        images_N: downsampled images by a factor of N
            ...
        poses_bounds.npy: currently unused
        sparse:
            0:
                cameras.bin
                images.bin
                points3D.bin
    """

    def __init__(
        self,
        colmap_directory_path: str,
        device: torch.device,
        downsample_factor: int,
        config: SplatConfig,
    ) -> None:
        super().__init__(config)

        self.colmap_directory_path = colmap_directory_path
        self.device = device
        self.downsample_factor = downsample_factor

        # load sparse points
        points_path = os.path.join(colmap_directory_path, "sparse", "0", "points3D.bin")
        sparse_points = read_points3D_binary(points_path)
        num_points = len(sparse_points)

        self.xyz = torch.zeros(num_points, 3)
        self.rgb = torch.zeros(num_points, 3)
        row = 0
        for _, point in sparse_points.items():
            self.xyz[row] = torch.tensor(point.xyz, dtype=torch.float32)
            self.rgb[row] = torch.tensor(
                point.rgb / 255.0 / 0.28209479177387814, dtype=torch.float32
            )
            row += 1
#ここから変更
         #images.binのパスを作る　"colmap_directory_path(bicycle~~)/sparse/0/images.bin"
         #image.bin.nameと一致している画像がimage_pathになり，image=cv2.imreadで読み込まれて，self.imageに格納
        image_info_path = os.path.join(colmap_directory_path, "sparse", "0", "images.bin")
        self.image_info = read_images_binary(image_info_path)

        #出力フォルダを作成 
        current_dir = Path(__file__).resolve().parent
        output_dir = current_dir.parent/"processed_poison"
        output_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        #image.bin.nameをimage_pathに
        #元画像の属性(psnr,ssim計算ではこちらを用いる)
        self.images_origin = []
        #フィルタ後画像の属性(最適化はこちらを基準にする)
        self.images = []
        #add clean
        self.images_clean =[]

#準備
        listy=[]
        total_filter_count = 0  # 全データセットでのフィルタ適用回数
        total_elements_count = 0  # 全データセットでの総要素数
        iv_avg_all = []  # 全データセットのIV_AVG値を格納

        for _, image_info in self.image_info.items():
            #image_pathを作る
            image_path = os.path.join(
                colmap_directory_path,
                f"images_{self.downsample_factor}",
                image_info.name,
            )
            #image_pathから読んだ入力にフィルタをかける
            imageBGR = cv2.imread(image_path)
            image = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
            #RGB画像(numpy)で計算
            filtered_image, listx,filter_count, total_elements = divide_calc(image,self.config.IV_AVG_THRESHOLD,self.config.DIVISIONS)
#情報の保持
            total_filter_count += filter_count
            total_elements_count += total_elements
            # IV_AVGの値を収集
            # iv_avg_all.extend([iv.item() for iv in listx])
            iv_avg_all.extend(listx) 
            listy.extend(listx)

            #output_pathを定義/保存
            output_path = os.path.join(output_dir, image_info.name)#input\name -> outdir\name
            filtered_image_BGR = cv2.cvtColor(filtered_image, cv2.COLOR_RGB2BGR) #cv.imwriteはBGRが前提
            cv2.imwrite(output_path, filtered_image_BGR) #パスに書き込み

            #output_dirに保存したフィルタ済み画像output_pathを読み込み 
            filtered_image = cv2.imread(output_path)
            filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)

#add clean
            # load clean image
            image_path2 = os.path.join(
                colmap_directory_path,
                f"images_{self.downsample_factor}clean",
                image_info.name,
            )
            image2 = cv2.imread(image_path2)
            image_clean = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)


            # load transform
            camera_T_world = torch.eye(4)
            camera_T_world[:3, :3] = torch.tensor(qvec2rotmat(image_info.qvec), dtype=torch.float32)
            camera_T_world[:3, 3] = torch.tensor(image_info.tvec, dtype=torch.float32)

            #self.imagesにfiltered_imageを追加
            self.images.append(
                Image(
                    image=torch.from_numpy(filtered_image).to(torch.uint8).to(self.device),
                    camera_id=image_info.camera_id,
                    camera_T_world=camera_T_world.to(self.device),
                )
            )
            #self.images_originに元画像(image)を追加
            self.images_origin.append(
                Image(
                    image=torch.from_numpy(image).to(torch.uint8).to(self.device),
                    camera_id=image_info.camera_id,
                    camera_T_world=camera_T_world.to(self.device),
                )
            )
#add
            self.images_clean.append(
                Image(
                    image=torch.from_numpy(image_clean).to(torch.uint8).to(self.device),
                    camera_id=image_info.camera_id,
                    camera_T_world=camera_T_world.to(self.device),
                )
            )
        # 統計情報を計算
        cdf_results = calculate_cdf(iv_avg_all)
        mean = cdf_results["mean"]
        std_dev = cdf_results["std_dev"]
        thresholds = cdf_results["thresholds"]

        # 統計情報を出力
        print(f"\niv_avgの平均値は {(sum(listy) / len(listy)):.4f}")
        print(f"iv_avgの最大値は{max(listy):.4f},最小値は{min(listy):.4f}")
        print(f"iv_avgの中央値は{statistics.median(listy)}")

        print(f"\nIV_AVGの統計情報:")
        print(f"- 平均値: {mean:.4f}")
        print(f"- 標準偏差: {std_dev:.4f}")
        print(f"- 上位5%閾値: {thresholds['top_5_percent']:.4f}")
        print(f"- 上位10%閾値: {thresholds['top_10_percent']:.4f}")
        print(f"- 下位5%閾値: {thresholds['bottom_5_percent']:.4f}")
        print(f"- 下位10%閾値: {thresholds['bottom_10_percent']:.4f}")

        #設定を表示
        print(f"\nthreshold={self.config.IV_AVG_THRESHOLD},division={self.config.DIVISIONS}")
        print(f"フィルタ適用: {total_filter_count}/{total_elements_count} 要素")
    
#ここまで変更
        # load cameras
        cameras_path = os.path.join(colmap_directory_path, "sparse", "0", "cameras.bin")
        cameras = read_cameras_binary(cameras_path)

        self.cameras = {}
        for camera_id, camera in cameras.items():
            K = torch.zeros((3, 3), dtype=torch.float32, device=self.device)
            if camera.model == "SIMPLE_PINHOLE":
                # colmap params [f, cx, cy]
                K[0, 0] = camera.params[0] / float(self.downsample_factor)
                K[1, 1] = camera.params[0] / float(self.downsample_factor)
                K[0, 2] = camera.params[1] / float(self.downsample_factor)
                K[1, 2] = camera.params[2] / float(self.downsample_factor)
                K[2, 2] = 1.0
            elif camera.model == "PINHOLE":
                # colmap params [fx, fy, cx, cy]
                K[0, 0] = camera.params[0] / float(self.downsample_factor)
                K[1, 1] = camera.params[1] / float(self.downsample_factor)
                K[0, 2] = camera.params[2] / float(self.downsample_factor)
                K[1, 2] = camera.params[3] / float(self.downsample_factor)
                K[2, 2] = 1.0
            else:
                raise NotImplementedError("Only Pinhole and Simple Pinhole cameras are supported")

            self.cameras[camera_id] = Camera(
                width=self.images[0].image.shape[1],
                height=self.images[0].image.shape[0],
                K=K,
            )
