import argparse
import numpy as np
import os

from eval.eval_utils import eval_visible_on_Housecat6D


if __name__ == "__main__":

    parser = argparse.ArgumentParser('UOIS CenterMask', add_help=False)

    # model config   
    parser.add_argument("--config-file", 
        default="./configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml", 
        # default="./configs/R50_rgb_mlc_occatmask_hom_concat.yaml", 
        metavar="FILE", help="path to config file")
    parser.add_argument("--gpu", type=str, default="0", help="GPU id")
    parser.add_argument(
        "--use-cgnet",
        action="store_true",
        help="Use foreground segmentation model to filter our background instances or not"
    )
    parser.add_argument(
        "--cgnet-weight-path",
        type=str,
        default="./foreground_segmentation/rgbd_fg.pth",
        help="path to forground segmentation weight"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/media/user/data1/dataset/Housecat6D",
        help="path to the Housecat6D dataset"
    )
    parser.add_argument(
        "--result-save-root",
        type=str,
        default="/media/user/data1/rcao/result/uois/Housecat6D/uoais-net_rgbd_cgnet_mask",
        help="path to save inference result"
    )

    args = parser.parse_args()
    print(args)
    # args.use_cgnet = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    eval_visible_on_Housecat6D(args)
    