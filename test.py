import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from thop import profile, clever_format
from PIL import Image
from torch.utils.data import DataLoader
from utils.dataset import get_loader
from model import myModel
from utils.metrics import Evaluator

def save_pred_only(pred, fn, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    pred = np.clip(pred, 0.0, 1.0)
    pred_uint8 = (pred * 255.0).round().astype(np.uint8)

    for i in range(pred_uint8.shape[0]):
        img = Image.fromarray(pred_uint8[i])

        name = fn[i]
        if isinstance(name, (list, tuple)):
            name = name[0]
        name = os.path.basename(str(name))

        out_path = os.path.join(save_dir, name)
        img.save(out_path)


def pad_to_multiple(x, base=16, mode="reflect"):
    _, _, h, w = x.shape
    ph = (base - h % base) % base
    pw = (base - w % base) % base
    x_pad = F.pad(x, (0, pw, 0, ph), mode=mode)
    return x_pad, (h, w)


def pad_collate(batch, base=16, pad_mode="reflect"):
    xs, ys, fns = zip(*batch)
    sizes = [(x.shape[1], x.shape[2]) for x in xs]

    Hmax = max(h for h, w in sizes)
    Wmax = max(w for h, w in sizes)

    Hpad = ((Hmax + base - 1) // base) * base
    Wpad = ((Wmax + base - 1) // base) * base

    x_pad_list, y_pad_list = [], []
    for x, y in zip(xs, ys):
        h, w = x.shape[1], x.shape[2]
        pad_h = Hpad - h
        pad_w = Wpad - w
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
        y_pad = F.pad(y, (0, pad_w, 0, pad_h), mode=pad_mode)
        x_pad_list.append(x_pad)
        y_pad_list.append(y_pad)

    return torch.stack(x_pad_list, 0), torch.stack(y_pad_list, 0), list(fns), sizes


class Tester(object):
    def __init__(self, args):
        self.args = args
        self.evaluator = Evaluator()

        self.deep_model = myModel(in_channels=3, feature_channels=32, use_white_balance=True)

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
            raise RuntimeError(f"=> no checkpoint found at '{args.ckpt}'")

        self.deep_model = self.deep_model.to("cuda")
        self.deep_model.eval()

        if args.dataset == "EUVP-Scenes":
            args.test_root = "D:/underwater/EUVP/scenes/test/"
            args.resize = False
            args.datasize = 256
        elif args.dataset == "UIEB":
            args.test_root = "D:/underwater/UIEB/test/"
            args.resize = False
            args.datasize = 256
        elif args.dataset == "UFO":
            args.test_root = "D:/underwater/UFO/test/"
            args.resize = False
            args.datasize = 256
        elif args.dataset == "LSUI":
            args.test_root = "D:/underwater/LSUI/test/"
            args.resize = False
            args.datasize = 256

        gl = get_loader(
            self.args.test_root,
            self.args.test_batch_size,
            self.args.datasize,
            train=False,
            resize=args.resize,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
        )

        if hasattr(gl, "dataset"):
            dataset = gl.dataset
        else:
            dataset = gl

        if self.args.test_batch_size > 1:
            self.dataloader = DataLoader(
                dataset,
                batch_size=self.args.test_batch_size,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
                collate_fn=lambda b: pad_collate(b, base=16, pad_mode="reflect"),
            )
        else:
            self.dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
            )

    def testing(self):
        self.evaluator.reset()
        torch.cuda.empty_cache()

        save_root = os.path.dirname(self.args.ckpt)
        save_dir = os.path.join(save_root, "pred")
        os.makedirs(save_dir, exist_ok=True)

        total_infer_time = 0.0
        total_images = 0
        with torch.no_grad():
            loop = tqdm(enumerate(self.dataloader), total=len(self.dataloader), leave=False)
            for _, batch in loop:

                if self.args.test_batch_size > 1:
                    x, label, fn, sizes = batch
                    x = x.cuda(non_blocking=True)

                    torch.cuda.synchronize()
                    start_time = time.time()

                    pred = self.deep_model(x)

                    torch.cuda.synchronize()
                    infer_time = time.time() - start_time

                    total_infer_time += infer_time
                    total_images += x.size(0)
                    pred = torch.clamp(pred, 0.0, 1.0)

                    pred_list, label_list = [], []
                    for i, (h, w) in enumerate(sizes):
                        pred_list.append(pred[i, :, :h, :w])
                        label_list.append(label[i, :, :h, :w])

                    pred = torch.stack(pred_list, 0)
                    label = torch.stack(label_list, 0)

                    pred_np = pred.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
                    label_np = label.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)

                    self.evaluator.evaluation(pred_np, label_np)
                    save_pred_only(pred_np, fn, save_dir)

                else:
                    x, label, fn = batch
                    x = x.cuda(non_blocking=True)

                    x_pad, (h0, w0) = pad_to_multiple(x, base=16, mode="reflect")

                    torch.cuda.synchronize()
                    start_time = time.time()

                    pred = self.deep_model(x_pad)

                    torch.cuda.synchronize()
                    infer_time = time.time() - start_time

                    total_infer_time += infer_time
                    total_images += 1
                    pred = pred[:, :, :h0, :w0]
                    pred = torch.clamp(pred, 0.0, 1.0)

                    pred_np = pred.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
                    label_np = label.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)

                    self.evaluator.evaluation(pred_np, label_np)
                    save_pred_only(pred_np, fn, save_dir)

                loop.set_description("[Testing]")

        ssim_, psnr_ = self.evaluator.getMean()
        avg_runtime = total_infer_time / total_images
        fps = total_images / total_infer_time
        print(
            "[Testing] SSIM: %.4f, PSNR: %.4f, Runtime: %.6f sec/img, FPS: %.2f"
            % (ssim_, psnr_, avg_runtime, fps)
        )

        with open(os.path.join(save_root, "result.txt"), "w") as f:
            f.write("[Testing] SSIM: %.4f, PSNR: %.4f" % (ssim_, psnr_))

        dummy = torch.randn(1, 3, 256, 256).cuda()
        flops, params = profile(self.deep_model, inputs=(dummy,))
        flops, params = clever_format([flops, params], "%.3f")
        print(f"Params: {params}, FLOPs: {flops}")

        model_info = {
            "params": params,
            "flops": flops,
            "ssim": "%.4f" % ssim_,
            "psnr": "%.4f" % psnr_,
        }
        return model_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="UEIB")
    parser.add_argument("--test_batch_size", type=int, default=8)
    args = parser.parse_args()

    tester = Tester(args)

    start = time.time()
    model_info = tester.testing()
    end = time.time()

    print("Testing time:", end - start, "sec")
    model_info["time"] = end - start


if __name__ == "__main__":
    main()