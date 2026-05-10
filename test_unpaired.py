import argparse
import os
import time
import numpy as np
import torch
from tqdm import tqdm
from thop import profile, clever_format
from PIL import Image
from utils.dataset import get_loader
from model import myModel
from utils.metrics import Evaluator


class Tester(object):
    def __init__(self, args):
        self.args = args

        self.deep_model = myModel(
            in_channels=3, feature_channels=32, use_white_balance=True
        )

        self.evaluator = Evaluator(no_ref=True)

        if os.path.isfile(args.ckpt):
            checkpoint = torch.load(args.ckpt, map_location="cpu")
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.ckpt))

        self.deep_model = self.deep_model.to("cuda")
        self.deep_model.eval()

        if args.dataset == "C60":
            args.test_root = "D:/underwater/C60/"
        elif args.dataset == "RUIE":
            args.test_root = "D:/underwater/RUIE/"
        elif args.dataset == "U45":
            args.test_root = "D:/underwater/U45/"
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        self.dataloader = get_loader(
            self.args.test_root,
            self.args.test_batch_size,
            256,
            train=False,
            resize=False,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            non_ref=True,
        )

    def _get_save_dir(self):

        ckpt_dir = os.path.dirname(self.args.ckpt)
        save_dir = os.path.join(ckpt_dir, "pred", self.args.dataset)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def testing(self):
        self.evaluator.reset()
        torch.cuda.empty_cache()

        save_dir = self._get_save_dir()

        with torch.no_grad():
            loop = tqdm(enumerate(self.dataloader), total=len(self.dataloader), leave=False)
            for _, (x, fn) in loop:
                x = x.to("cuda")

                pred = self.deep_model(x)
                pred = torch.clamp(pred, 0.0, 1.0)

                pred_np = pred.detach().cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)

                self.evaluator.evaluation(pred_np, None)

                for i in range(pred_np.shape[0]):
                    out_img = (pred_np[i] * 255).astype(np.uint8)
                    out_name = fn[i] + ".png"
                    Image.fromarray(out_img).save(os.path.join(save_dir, out_name))

                loop.set_description("[Testing]")

        means = self.evaluator.getMean()
        if isinstance(means, (list, tuple)):
            if len(means) >= 2:
                niqe_, uciqe_ = means[0], means[1]
                print("[Testing] NIQE: %.4f, UCIQE: %.4f" % (niqe_, uciqe_))
            else:
                print("[Testing] getMean() returned:", means)
        else:
            print("[Testing] getMean() returned:", means)

        dummy = torch.randn(1, 3, 256, 256).cuda()
        flops, params = profile(self.deep_model, inputs=(dummy,))
        flops, params = clever_format([flops, params], "%.3f")
        print(f"Params: {params}, FLOPs: {flops}")

        return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="EUVP")
    parser.add_argument("--test_batch_size", type=int, default=8)

    args = parser.parse_args()

    tester = Tester(args)

    start = time.time()
    tester.testing()
    end = time.time()
    print("Testing time:", end - start, "sec")


if __name__ == "__main__":
    main()
