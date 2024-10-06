import torch
import torch.nn as nn
import os

from ssd.config import cfg
from inference.ssd_xmem_infer_detector import SSDXMemDetector
from ssd.modeling.backbone.common import C3
from ssd.modeling.backbone.darknet import BaseConv
from ssd.modeling.box_head import build_box_head
from torch.onnx import export


class PANHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        width, depth = cfg.MODEL.BACKBONE.WIDTH, cfg.MODEL.BACKBONE.DEPTH
        in_channels = [256, 512, 1024]

        self.box_head = build_box_head(cfg)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            round(in_channels[2] * width), round(in_channels[1] * width), 1, 1
        )
        self.C3_p4 = C3(
            c1=round(2 * in_channels[1] * width),
            c2=round(in_channels[1] * width),
            n=round(3 * depth),
            shortcut=False,
        )  # cat

        self.reduce_conv1 = BaseConv(
            round(in_channels[1] * width), round(in_channels[0] * width), 1, 1
        )
        self.C3_p3 = C3(
            c1=round(2 * in_channels[0] * width),
            c2=round(in_channels[0] * width),
            n=round(3 * depth),
            shortcut=False,
        )

        # bottom-up conv
        self.bu_conv2 = BaseConv(
            round(in_channels[0] * width), round(in_channels[0] * width), 3, 2
        )
        self.C3_n3 = C3(
            c1=round(2 * in_channels[0] * width),
            c2=round(in_channels[1] * width),
            n=round(3 * depth),
            shortcut=False,
        )

        # bottom-up conv
        self.bu_conv1 = BaseConv(
            round(in_channels[1] * width), round(in_channels[1] * width), 3, 2
        )
        self.C3_n4 = C3(
            c1=round(2 * in_channels[1] * width),
            c2=round(in_channels[2] * width),
            n=round(3 * depth),
            shortcut=False,
        )

    def forward(self, x2, x1, x0):
        fpn_out0 = self.lateral_conv0(x0)  # 2048->1024/32
        f_out0 = self.upsample(fpn_out0)  # 1024/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 1024->2048/16
        f_out0 = self.C3_p4(f_out0)  # 2048->1024/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 1024->512/16
        f_out1 = self.upsample(fpn_out1)  # 512/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 512->1024/8
        pan_out2 = self.C3_p3(f_out1)  # 1024->512/8

        # try to add ASPP module to make it attention to global tiny object
        # pan_out2 = self.aspp(pan_out2)

        p_out1 = self.bu_conv2(pan_out2)  # 512->512/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 512->1024/16
        pan_out1 = self.C3_n3(p_out1)  # 1024->1024/16

        p_out0 = self.bu_conv1(pan_out1)  # 1024->1024/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 1024->2048/32
        pan_out0 = self.C3_n4(p_out0)  # 2048->2048/32

        outputs = (pan_out2, pan_out1, pan_out0)
        detections, _ = self.box_head(outputs)
        boxes, scores, labels = detections

        return boxes, scores, labels


def get_pan_model(cfg, weight_path, device):
    weights = torch.load(weight_path)["model"]
    pan_model = PANHead(cfg).to(device)
    new_state_dict = {}

    for k, v in weights.items():
        if "pan" in k:
            new_state_dict[k.replace("backbone.pan.", "")] = v

        if "box_head.predictor" in k:
            # new_k = "box_predictor." + k[19:]
            new_state_dict[k] = v

    missing_keys, unexpected_keys = pan_model.load_state_dict(
        new_state_dict, strict=True
    )

    return pan_model


class SSDXMemExporter:
    def __init__(self, cfg, weight_path, device="cpu", output_path="./onnx_output/"):
        model = SSDXMemDetector(cfg)
        missing_keys, unexpected_keys = model.load_state_dict(
            torch.load(weight_path, map_location=device)["model"], strict=False
        )
        model.eval()
        model.to(device)

        self.width = cfg.MODEL.BACKBONE.WIDTH
        self.depth = cfg.MODEL.BACKBONE.DEPTH

        self.pan = get_pan_model(cfg, weight_path, device)
        self.pan.eval()

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        self.device = device

        self.backbone = model.backbone.darknet
        self.stam = model.backbone.stam
        self.ltam = model.ltam
        # self.pan = model.backbone.pan

        self.output_path = output_path

    def export_backbone(self):
        print("[I] Exporting Backbone onnx model...")
        dummy_input = torch.randn((1, 3, 320, 320), device=self.device)
        export(
            self.backbone,
            dummy_input,
            os.path.join(self.output_path, "backbone_m.onnx"),
            input_names=["img"],
            output_names=["feat_40", "feat_20", "feat_10"],
            opset_version=14,
            verbose=True,
        )
        print("[I] Backbone onnx model exported.")

    def export_stam(self):
        print("[I] Exporting STAM onnx model...")
        dummy_input = torch.randn(
            (1, 3, round(1024 * self.width), 10, 10), device=self.device, dtype=torch.float32
        )
        # dummy_input_x1 = torch.randn(
        #     (1, 3, round(512 * self.width), 20, 20), device=self.device, dtype=torch.float32
        # )
        # dummy_input_x2 = torch.randn(
        #     (1, 3, round(256 * self.width), 40, 40), device=self.device, dtype=torch.float32
        # )
        # pe_enc, pe_dec = self.stam.pe_enc, self.stam.pe_dec

        export(
            self.stam,
            dummy_input,
            os.path.join(self.output_path, "stam_m.onnx"),
            input_names=["x0"],
            output_names=["stam_x0"],
            opset_version=14,
            verbose=True,
        )
        print("[I] STAM onnx model exported.")

    def export_ltam(self):
        dummy_q_x = torch.randn((1, 1, round(1024 * self.width), 10, 10), device=self.device)
        dummy_enc_x = torch.randn((1, 3, round(1024 * self.width), 10, 10), device=self.device)

        print("[I] Exporting LTAM onnx model...")
        export(
            self.ltam,
            (dummy_enc_x, dummy_q_x),
            os.path.join(self.output_path, "ltam_m.onnx"),
            input_names=["ltam_enc", "ltam_q"],
            output_names=["ltam_mem", "ltam_lt"],
            opset_version=14,
            verbose=True,
        )
        print("[I] LTAM onnx model exported.")

    def export_pan(self):
        dummy_input_x2 = torch.randn((1, round(256 * self.width), 40, 40), device=self.device)
        dummy_input_x1 = torch.randn((1, round(512 * self.width), 20, 20), device=self.device)
        dummy_input_x0 = torch.randn((1, round(1024 * self.width), 10, 10), device=self.device)
        print("[I] Exporting PAN onnx model...")
        export(
            self.pan,
            (dummy_input_x2, dummy_input_x1, dummy_input_x0),
            os.path.join(self.output_path, "pan_m.onnx"),
            input_names=["lt_40", "lt_20", "lt_10"],
            output_names=["boxes", "scores", "labels"],
            opset_version=14,
            verbose=True,
        )
        print("[I] PAN onnx model exported.")

    def __call__(self, *args, **kwargs):
        self.export_backbone()
        self.export_stam()
        self.export_ltam()
        self.export_pan()


if __name__ == "__main__":
    config_path = "./configs/resnet/dark/darknet_ssd320_dark_mem.yaml"
    weight_path = "./outputs/1103_m/last_iteration_model.pth"
    cfg.merge_from_file(config_path)
    cfg.freeze()
    print("Loaded configuration file {}".format(config_path))
    with open(config_path, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    exporter = SSDXMemExporter(cfg, weight_path, device="cpu")
    exporter()
    # exporter.export_stam()
    # exporter.export_ltam()
    # exporter.export_pan()
