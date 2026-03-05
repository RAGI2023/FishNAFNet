import os
import cv2
import numpy as np
from functools import lru_cache
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import img2tensor, padding
from basicsr.data.equirect_utils import equirect_to_fisheye_ucm


@lru_cache(maxsize=4)
def _read_image_cached(path):
    """Per-worker LRU cache for equirectangular images (RGB uint8)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class FisheyeDeblurDataset(data.Dataset):
    """Fisheye deblur dataset for NAFNet.

    Loads paired equirectangular (360°) images and projects them on-the-fly
    to fisheye images using the UCM (Unified Camera Model). Both blurry (lq)
    and sharp (gt) images are projected with identical parameters to guarantee
    spatial alignment.

    Each equirectangular image produces 4 fisheye views (front/right/back/left),
    so dataset length = num_image_pairs * 4.

    Config keys:
        dataroot_lq (str): Folder with blurry equirectangular images.
        dataroot_gt (str): Folder with sharp equirectangular images.

        # Fisheye UCM parameters (no jitter; fixed for both lq and gt)
        out_w (int):        Output fisheye width in pixels.  Default: 512
        out_h (int):        Output fisheye height in pixels. Default: 512
        xi (float):         UCM mirror parameter.            Default: 0.9
        f_pix (float|None): Focal length in pixels.          Default: None
        fov_diag_deg (float): Diagonal FOV (deprecated fallback). Default: 130
        mask_mode (str):    'inscribed'|'diagonal'|'none'.   Default: 'inscribed'

        # Augmentation (training phase only)
        gt_size (int|None): Random crop patch size. None = no crop.
        use_flip (bool):    Horizontal flip augmentation.    Default: True
        use_rot (bool):     Rotation augmentation.           Default: True

        scale (int):  Restoration scale factor (1 for deblur). Default: 1
        phase (str):  'train' or 'val'.
    """

    VIEWS = [
        ("front", np.array([0.0,  0.0,  1.0], dtype=np.float32)),
        ("right", np.array([1.0,  0.0,  0.0], dtype=np.float32)),
        ("back",  np.array([0.0,  0.0, -1.0], dtype=np.float32)),
        ("left",  np.array([-1.0, 0.0,  0.0], dtype=np.float32)),
    ]

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.lq_folder = opt['dataroot_lq']
        self.gt_folder = opt['dataroot_gt']

        # Fisheye projection parameters
        self.out_w = int(opt.get('out_w', 512))
        self.out_h = int(opt.get('out_h', 512))
        self.xi = float(opt.get('xi', 0.9))
        self.f_pix = float(opt['f_pix']) if opt.get('f_pix') is not None else None
        self.fov = float(opt.get('fov_diag_deg', 130))
        self.mask_mode = opt.get('mask_mode', 'inscribed')

        # Scan and pair images by sorted filename
        exts = ('.png', '.jpg', '.jpeg')
        lq_files = sorted(f for f in os.listdir(self.lq_folder) if f.lower().endswith(exts))
        gt_files = sorted(f for f in os.listdir(self.gt_folder) if f.lower().endswith(exts))

        assert len(lq_files) == len(gt_files), (
            f"File count mismatch: {len(lq_files)} lq vs {len(gt_files)} gt in "
            f"{self.lq_folder} / {self.gt_folder}"
        )
        assert lq_files == gt_files, (
            "Filename mismatch between lq and gt folders. "
            "Files must have identical names."
        )

        self.pairs = [
            (os.path.join(self.lq_folder, f), os.path.join(self.gt_folder, f))
            for f in lq_files
        ]

        # Optionally limit number of image pairs (useful for fast debug validation)
        limit = opt.get('val_limit', None)
        if limit is not None:
            self.pairs = self.pairs[:int(limit)]

    def __len__(self):
        return len(self.pairs) * len(self.VIEWS)

    def _project(self, img_rgb, base_dir):
        """Project equirectangular RGB uint8 image to fisheye, return RGB uint8."""
        ucm_kwargs = dict(xi=self.xi, mask_mode=self.mask_mode)
        if self.f_pix is not None:
            ucm_kwargs['f_pix'] = self.f_pix
        else:
            ucm_kwargs['fov_diag_deg'] = self.fov

        out = equirect_to_fisheye_ucm(
            img_rgb,
            out_w=self.out_w,
            out_h=self.out_h,
            base_dir=base_dir,
            yaw_deg=0.0,
            pitch_deg=0.0,
            roll_deg=0.0,
            jitter_cfg=None,
            **ucm_kwargs,
        )
        return out  # HWC RGB uint8

    def __getitem__(self, index):
        img_idx = index // len(self.VIEWS)
        view_idx = index % len(self.VIEWS)

        lq_path, gt_path = self.pairs[img_idx]
        view_name, base_dir = self.VIEWS[view_idx]

        # Load equirectangular images (cached per worker to avoid 4x redundant reads)
        img_lq = _read_image_cached(lq_path)
        img_gt = _read_image_cached(gt_path)

        # Project both with identical parameters
        img_lq = self._project(img_lq, base_dir).astype(np.float32) / 255.0  # HWC RGB [0,1]
        img_gt = self._project(img_gt, base_dir).astype(np.float32) / 255.0  # HWC RGB [0,1]

        # Training augmentation
        if self.opt['phase'] == 'train':
            gt_size = self.opt.get('gt_size', None)
            scale = self.opt.get('scale', 1)
            if gt_size is not None:
                img_gt, img_lq = padding(img_gt, img_lq, gt_size)
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            img_gt, img_lq = augment(
                [img_gt, img_lq],
                self.opt.get('use_flip', True),
                self.opt.get('use_rot', True),
            )

        # HWC RGB float32 -> CHW tensor (already RGB, no bgr2rgb conversion)
        img_lq, img_gt = img2tensor([img_lq, img_gt], bgr2rgb=False, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': f"{lq_path}:{view_name}",
            'gt_path': f"{gt_path}:{view_name}",
        }
