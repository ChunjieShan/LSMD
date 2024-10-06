import math
import torch
from mmcv.cnn import ConvModule
from torch import nn

from ssd.modeling.backbone import build_backbone
from ssd.modeling.box_head import build_box_head
from ssd.modeling.memory.modules import KeyProjection, SequenceConv, MemoryModule
from ssd.modeling.stf import STF
# from ssd.modeling.swin import LTAM
from ssd.modeling.temporal_modules import LTAM


class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.neck = STF(dim=1024, depth=1, heads=2, mlp_dim=256, pe_dim=(10, 10), block_size=(2, 2),
                        frame_len=cfg.DATASETS.FRAME_LENGTH)
        self.box_head = build_box_head(cfg)

    def forward(self, images, targets=None, orig_targets=None):
        features = self.backbone(images)
        features = self.neck(features[0])
        features = [features]

        detections, detector_losses = self.box_head(features, targets)
        if self.training:
            return detector_losses
        return detections


class SSDXMemDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.neck = STF(dim=1024, depth=1, heads=2, mlp_dim=256, pe_dim=(10, 10), block_size=(2, 2),
                        frame_len=cfg.DATASETS.FRAME_LENGTH)
        self.box_head = build_box_head(cfg)
        self.key_proj = KeyProjection(in_channels=1024,
                                      out_channels=1024)

    def encode_key_and_value(self, aggregated_feat):
        if isinstance(aggregated_feat, list):
            aggregated_feat = aggregated_feat[0]

        key, _, _ = self.key_proj(aggregated_feat)
        return [key, aggregated_feat]

    def forward(self, prev_images, curr_images, targets=None):
        with torch.no_grad():
            prev_feat = self.forward_prev_clips(prev_images)
        torch.cuda.empty_cache()
        curr_agg_feat = self.forward_curr_clips(curr_images, prev_feat)

        if isinstance(curr_agg_feat, torch.Tensor):
            curr_agg_feat = [curr_agg_feat]

        detections, detector_losses = self.box_head(curr_agg_feat, targets)
        if self.training:
            return detector_losses
        return curr_agg_feat, detections

    def forward_backbone_and_neck(self, images):
        feats_list = self.backbone(images)
        features = []

        for feat in feats_list:
            features.append(self.neck(feat))

        return features

    def forward_prev_clips(self, prev_images):
        ndim = prev_images.ndim
        assert ndim == 4, \
            "Expected dimension of previous images should be 4, got {}".format(
                ndim)
        with torch.no_grad():
            # prev_features = self.backbone(prev_images)
            prev_features = self.forward_backbone_and_neck(prev_images)

        return prev_features

    def forward_curr_clips(self,
                           curr_images,
                           prev_feat):
        ndim = curr_images.ndim
        assert ndim == 4, \
            "Expected dimension of previous images should be 4, got {}".format(
                ndim)

        with torch.no_grad():
            curr_feat = self.forward_backbone_and_neck(curr_images)
        # Backbone and STF won't be updated
        # with torch.no_grad():
        #     curr_feat = self.forward_backbone_and_neck(curr_images)

        if prev_feat is not None:
            if isinstance(prev_feat, list):
                prev_feat = prev_feat[0]

            prev_key, prev_value = self.encode_key_and_value(prev_feat)
            curr_key, curr_value = self.encode_key_and_value(curr_feat)
            memory_readout = self.readout(
                prev_key, prev_value, curr_key, curr_value)

            return memory_readout
        else:
            detections, detector_losses = self.box_head(curr_feat, None)
            return curr_feat, detections

    def readout(self,
                prev_key,
                prev_value,
                curr_key,
                curr_value):
        affinity = self._get_affinity(prev_key, curr_key)
        B, CV, H, W = curr_value.shape
        mo = prev_value.view(B, CV, H * W)
        mem = torch.bmm(mo, affinity)  # Weighted-sum B, CV, HW
        mem = mem.view(B, CV, H, W)

        mem_out = torch.cat([mem, curr_value], dim=1)

        return mem_out

    @staticmethod
    def _get_affinity(mk: torch.Tensor,
                      qk: torch.Tensor):
        assert (mk.ndim == 4) and (qk.ndim == 4) and (mk.ndim == qk.ndim), \
            "memory key or query key dimension error!"

        B, CK, H, W = mk.shape
        mk = mk.flatten(start_dim=2)
        qk = qk.flatten(start_dim=2)

        # See supplementary material
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk

        affinity = (2 * ab - a_sq) / math.sqrt(CK)  # B, THW, HW

        # softmax operation; aligned the evaluation style
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum

        return affinity

    @staticmethod
    def _freeze_backbone_weight(backbone: torch.nn.Module):
        for i, (name, param) in enumerate(backbone.named_parameters()):
            if "resnet" in name:
                param.requires_grad = False


class SSDTMADetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.in_channels = 1024
        self.neck = STF(dim=1024, depth=1, heads=2, mlp_dim=256, pe_dim=(10, 10), block_size=(2, 2),
                        frame_len=cfg.DATASETS.FRAME_LENGTH)
        self.box_head = build_box_head(cfg)
        self.norm_cfg = dict(type='BN', requires_grad=True)
        self.conv_cfg = None
        self.act_cfg = dict(type='ReLU')

        value_channels = 512
        key_channels = 256
        sequence_num = 1

        self.memory_key_conv = nn.Sequential(
            SequenceConv(self.in_channels, key_channels, 1, sequence_num,
                         self.conv_cfg, self.norm_cfg, self.act_cfg),
            SequenceConv(key_channels, key_channels, 3, sequence_num,
                         self.conv_cfg, self.norm_cfg, self.act_cfg)
        )

        self.memory_value_conv = nn.Sequential(
            SequenceConv(self.in_channels, value_channels, 1, sequence_num,
                         self.conv_cfg, self.norm_cfg, self.act_cfg),
            SequenceConv(value_channels, value_channels, 3, sequence_num,
                         self.conv_cfg, self.norm_cfg, self.act_cfg)
        )

        self.query_key_conv = nn.Sequential(
            ConvModule(
                self.in_channels,
                key_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                key_channels,
                key_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )

        self.query_value_conv = nn.Sequential(
            ConvModule(
                self.in_channels,
                value_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                value_channels,
                value_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )

        self.memory_module = MemoryModule(matmul_norm=False)
        self.bottleneck = ConvModule(
            value_channels * 2,
            2048,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

    def encode_memory_kv(self, aggregated_feat):
        if isinstance(aggregated_feat, list):
            aggregated_feat = aggregated_feat[0]

        key = self.memory_key_conv(aggregated_feat)
        value = self.memory_value_conv(aggregated_feat)
        return [key, value]

    def encode_query_kv(self, aggregated_feat):
        if isinstance(aggregated_feat, list):
            aggregated_feat = aggregated_feat[0]

        key = self.query_key_conv(aggregated_feat)
        value = self.query_value_conv(aggregated_feat)
        return [key, value]

    def forward(self, prev_images, curr_images, targets=None):
        with torch.no_grad():
            prev_feat = self.forward_prev_clips(prev_images)
        torch.cuda.empty_cache()
        curr_agg_feat = self.forward_curr_clips(curr_images, prev_feat)

        if isinstance(curr_agg_feat, torch.Tensor):
            curr_agg_feat = [curr_agg_feat]

        detections, detector_losses = self.box_head(curr_agg_feat, targets)
        if self.training:
            return detector_losses
        return curr_agg_feat, detections

    def forward_backbone_and_neck(self, images):
        with torch.no_grad():
            feats_list = self.backbone(images)
        features = []

        for feat in feats_list:
            features.append(self.neck(feat))

        return features

    def forward_prev_clips(self, prev_images):
        ndim = prev_images.ndim
        assert ndim == 4, \
            "Expected dimension of previous images should be 4, got {}".format(
                ndim)
        with torch.no_grad():
            # prev_features = self.backbone(prev_images)
            prev_features = self.forward_backbone_and_neck(prev_images)

        return prev_features

    def forward_curr_clips(self,
                           curr_images,
                           prev_feat):
        ndim = curr_images.ndim
        assert ndim == 4, \
            "Expected dimension of previous images should be 4, got {}".format(
                ndim)

        with torch.no_grad():
            curr_feat = self.forward_backbone_and_neck(curr_images)
        # Backbone and STF won't be updated
        # with torch.no_grad():
        #     curr_feat = self.forward_backbone_and_neck(curr_images)

        if prev_feat is not None:
            if isinstance(prev_feat, list):
                prev_feat = prev_feat[0]

            if prev_feat.ndim == 4:  # b(t) * c * h * w
                b, c, h, w = prev_feat.shape
                prev_feat = prev_feat.view(1, -1, c, h, w)  # t * b * c * h * w
            prev_key, prev_value = self.encode_memory_kv(prev_feat)
            curr_key, curr_value = self.encode_query_kv(curr_feat)
            memory_readout = self.memory_module(
                prev_key, prev_value, curr_key, curr_value)
            memory_readout = self.bottleneck(memory_readout)

            return memory_readout
        else:
            detections, detector_losses = self.box_head(curr_feat, None)
            return curr_feat, detections

    @staticmethod
    def _freeze_backbone_weight(backbone: torch.nn.Module):
        for i, (name, param) in enumerate(backbone.named_parameters()):
            if "resnet" in name:
                param.requires_grad = False


class SSDTMAPANDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.backbone.train(False)

        self.in_channels = 1024
        self.box_head = build_box_head(cfg)
        self.norm_cfg = dict(type='BN', requires_grad=True)
        self.conv_cfg = None
        self.act_cfg = dict(type='ReLU')

        self.value_channels = value_channels = 256
        self.key_channels = key_channels = 64
        sequence_num = 1
        # self.pan = PAN(in_channels=[512, 1024, 2048], width=1.0, depth=1.0, frame_len=cfg.DATASETS.FRAME_LENGTH)
        # self.pan.train()

        self.memory_key_conv = nn.Sequential(
            SequenceConv(self.in_channels, key_channels, 1, sequence_num,
                         self.conv_cfg, self.norm_cfg, self.act_cfg),
            SequenceConv(key_channels, key_channels, 3, sequence_num,
                         self.conv_cfg, self.norm_cfg, self.act_cfg)
        )

        self.memory_value_conv = nn.Sequential(
            SequenceConv(self.in_channels, value_channels, 1, sequence_num,
                         self.conv_cfg, self.norm_cfg, self.act_cfg),
            SequenceConv(value_channels, value_channels, 3, sequence_num,
                         self.conv_cfg, self.norm_cfg, self.act_cfg)
        )

        self.query_key_conv = nn.Sequential(
            ConvModule(
                self.in_channels,
                key_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                key_channels,
                key_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )

        self.query_value_conv = nn.Sequential(
            ConvModule(
                self.in_channels,
                value_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                value_channels,
                value_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )

        self.memory_module = MemoryModule(matmul_norm=False)
        self.bottleneck = ConvModule(
            value_channels * 2,
            1024,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

    def freeze_weights(self):
        self._freeze_darknet_weights()
        self._freeze_stam_weights()

    def _freeze_darknet_weights(self):
        for param in self.backbone.darknet.parameters():
            param.requires_grad = False

    def _freeze_stam_weights(self):
        for param in self.backbone.stam.parameters():
            param.requires_grad = False

    def encode_memory_kv(self, aggregated_feat):
        if isinstance(aggregated_feat, list):
            aggregated_feat = aggregated_feat[0]

        key = self.memory_key_conv(aggregated_feat)
        value = self.memory_value_conv(aggregated_feat)
        return [key, value]

    def encode_query_kv(self, aggregated_feat):
        if isinstance(aggregated_feat, list):
            aggregated_feat = aggregated_feat[0]

        key = self.query_key_conv(aggregated_feat)
        value = self.query_value_conv(aggregated_feat)
        return [key, value]

    def forward(self, prev_images, curr_images, targets=None):
        # with torch.no_grad():
        #     prev_feat = self.forward_prev_clips(prev_images)
        prev_feat = self.forward_prev_clips(prev_images)
        torch.cuda.empty_cache()
        curr_agg_feat = self.forward_curr_clips(curr_images, prev_feat)

        if isinstance(curr_agg_feat, torch.Tensor):
            curr_agg_feat = [curr_agg_feat]

        detections, detector_losses = self.box_head(curr_agg_feat, targets)
        if self.training:
            return detector_losses
        return curr_agg_feat, detections

    def forward_backbone(self, images):
        """

        :param images: torch.Tensor
        :return: Dict
        """
        # with torch.no_grad():
        #     features = self.backbone(images)
        features = self.backbone.forward_features(images)

        # feats_list = self.backbone(images)
        # features = []
        #
        # for feat in feats_list:
        #     features.append(self.neck(feat))

        return features

    def forward_stf(self, x):
        return self.backbone.forward_stf(x)

    def forward_backbone_stam(self, images):
        if images.ndim > 5:
            images = images.squeeze(1)
        xs = self.forward_backbone(images)
        xs["dark5"] = self.forward_attn(xs["dark5"])
        return xs

    def forward_tma(self, prev_feat, curr_feat):
        """

        :param prev_feat: previous video clip features
        :param curr_feat: current video clip features
        :return: aggregated enhanced current video clip features
        """
        if isinstance(prev_feat, dict):
            prev_x0 = prev_feat["dark5"]
        else:
            prev_x0 = prev_feat

        if isinstance(curr_feat, dict):
            curr_x0 = curr_feat["dark5"]
        else:
            curr_x0 = curr_feat

        if prev_x0.ndim == 4:  # b(t) * c * h * w
            b, c, h, w = prev_x0.shape
            prev_x0 = prev_x0.view(1, -1, c, h, w)  # t * b * c * h * w

        prev_key, prev_value = self.encode_memory_kv(prev_x0)
        curr_key, curr_value = self.encode_query_kv(curr_x0)
        memory_readout = self.memory_module(
            prev_key, prev_value, curr_key, curr_value)
        memory_readout = self.bottleneck(memory_readout)

        return memory_readout

    def forward_prev_clips(self, prev_images):
        ndim = prev_images.ndim
        assert ndim == 5, \
            "Expected dimension of previous images should be 5, got {}".format(
                ndim)
        # with torch.no_grad():
        #     prev_features = self.forward_backbone_and_neck(prev_images)
        # prev_features = self.backbone(prev_images)

        return self.forward_backbone_stam(prev_images)

    def forward_curr_clips(self,
                           curr_images,
                           prev_feat):
        ndim = curr_images.ndim
        assert ndim == 5, \
            "Expected dimension of previous images should be 5, got {}".format(
                ndim)

        # with torch.no_grad():
        #     curr_feat = self.forward_backbone_and_neck(curr_images)

        # current features with aggregated
        curr_feat = self.forward_backbone_stam(curr_images)
        if prev_feat is not None:
            curr_x0 = self.forward_tma(prev_feat, curr_feat)
            curr_feat["dark5"] = curr_x0
        else:
            neck_outputs = self.backbone.forward_pan(curr_feat)
            detections, detector_losses = self.box_head(neck_outputs, None)
            return curr_feat, detections

        neck_outputs = self.backbone.forward_pan(curr_feat)

        # Backbone and STF won't be updated
        # with torch.no_grad():
        #     curr_feat = self.forward_backbone_and_neck(curr_images)
        return neck_outputs

    @staticmethod
    def _freeze_backbone_weight(backbone: torch.nn.Module):
        for i, (name, param) in enumerate(backbone.named_parameters()):
            if "resnet" in name:
                param.requires_grad = False


class SSDLSTPANDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_prev_clips = cfg.SOLVER.NUM_PREV_CLIPS
        self.frame_len = cfg.DATASETS.FRAME_LENGTH
        self.backbone = build_backbone(cfg)

        self.in_channels = self.value_channels = 1024
        self.box_head = build_box_head(cfg)
        self.norm_cfg = dict(type='BN', requires_grad=True)
        self.conv_cfg = None
        self.act_cfg = dict(type='ReLU')
        self.ltam = LTAM(cfg)

    def freeze_weights(self):
        self._freeze_darknet_weights()
        self._freeze_stam_weights()
        # self._freeze_pan_weights()
        # self._freeze_head_weights()

    def _freeze_darknet_weights(self):
        for param in self.backbone.darknet.parameters():
            param.requires_grad = False

    def _freeze_stam_weights(self):
        for param in self.backbone.stam.parameters():
            param.requires_grad = False

    def _freeze_head_weights(self):
        for param in self.box_head.parameters():
            param.requires_grad = False

    def _freeze_pan_weights(self):
        for param in self.backbone.pan.parameters():
            param.requires_grad = False

    def forward(self, prev_images, curr_images, targets=None):
        # with torch.no_grad():
        #     prev_feat = self.forward_prev_clips(prev_images)
        if self.num_prev_clips == 6:
            prev_feat = self.forward_prev_6_clips(prev_images)
        elif self.num_prev_clips == 5:
            prev_feat = self.forward_prev_5_clips(prev_images)
        else:
            prev_feat = self.forward_prev_clips(prev_images)

        # prev_feat = torch.cat(prev_feat, dim=2)
        # prev_feat = self.memory_decoder(prev_feat)

        # torch.cuda.empty_cache()
        curr_agg_feat = self.forward_curr_clips(curr_images, prev_feat)

        if isinstance(curr_agg_feat, torch.Tensor):
            curr_agg_feat = [curr_agg_feat]

        detections, detector_losses = self.box_head(curr_agg_feat, targets)
        if self.training:
            return detector_losses
        return curr_agg_feat, detections

    def forward_backbone_attn(self, images):
        if images.ndim > 5:
            images = images.squeeze(1)
        xs = self.forward_backbone(images)

        if self.frame_len > 1:
            xs["dark5"] = self.forward_stam(xs["dark5"])
        else:
            xs["dark5"] = xs["dark5"].unsqueeze(1)

        return xs

    def forward_stam(self, x):
        return self.backbone.forward_stam(x)

    def forward_ltam(self, q_x, enc_x):
        # n, l, c = q_x.shape
        # kv_x = enc_x.flatten(1, 2)
        return self.ltam(enc_x, q_x)

    def forward_backbone(self, images):
        """

        :param images: torch.Tensor
        :return: Dict
        """
        features = self.backbone.forward_features(images)

        return features

    def forward_prev_clips(self, prev_clips):
        prev_clips = torch.chunk(
            prev_clips, chunks=prev_clips.shape[1], dim=1)

        prev_sts = []

        # with torch.no_grad():
        for prev_clip in prev_clips:
            prev_st = self.forward_backbone_attn(prev_clip)["dark5"]
            if prev_st.ndim < 5:
                prev_st = prev_st.unsqueeze(1)
            prev_sts.append(prev_st)

        return prev_sts

    def forward_prev_6_clips(self, prev_clips):
        prev_feats = self.forward_prev_clips(prev_clips=prev_clips)
        prev6_feat_st, prev5_feat_st, prev4_feat_st, prev3_feat_st, prev2_feat_st, prev1_feat_st = prev_feats
        # _, prev4_feat_lt = self.forward_ltam(prev4_feat_st, prev5_feat_st)
        # _, prev3_feat_lt = self.forward_ltam(prev3_feat_st, prev4_feat_lt)
        # _, prev2_feat_lt = self.forward_ltam(prev2_feat_st, prev3_feat_lt)
        # _, prev1_feat_lt = self.forward_ltam(prev1_feat_st, prev2_feat_lt)
        prev_clips_enc = torch.cat([prev6_feat_st, prev5_feat_st, prev4_feat_st], dim=1)
        memory_3, prev3_feat_lt = self.forward_ltam(prev3_feat_st, prev_clips_enc)
        memory_3 = self.conv3d_trans(memory_3.transpose(1, 2)).transpose(1, 2)

        prev_clips_enc = torch.cat([prev4_feat_st, prev3_feat_lt, memory_3], dim=1)
        memory_2, prev2_feat_lt = self.forward_ltam(prev2_feat_st, prev_clips_enc)
        memory_2 = self.conv3d_trans(memory_2.transpose(1, 2)).transpose(1, 2)

        prev_clips_enc = torch.cat([prev3_feat_lt, prev2_feat_lt, memory_2], dim=1)
        memory_1, prev1_feat_lt = self.forward_ltam(prev1_feat_st, prev_clips_enc)
        memory_1 = self.conv3d_trans(memory_1.transpose(1, 2)).transpose(1, 2)

        return prev2_feat_lt, prev1_feat_lt, memory_1

    def forward_prev_5_clips(self, prev_clips):
        prev_feats = self.forward_prev_clips(prev_clips=prev_clips)
        prev5_feat_st, prev4_feat_st, prev3_feat_st, prev2_feat_st, prev1_feat_st = prev_feats
        # _, prev4_feat_lt = self.forward_ltam(prev4_feat_st, prev5_feat_st)
        # _, prev3_feat_lt = self.forward_ltam(prev3_feat_st, prev4_feat_lt)
        # _, prev2_feat_lt = self.forward_ltam(prev2_feat_st, prev3_feat_lt)
        # _, prev1_feat_lt = self.forward_ltam(prev1_feat_st, prev2_feat_lt)
        prev_clips_enc = torch.cat([prev5_feat_st, prev4_feat_st, prev3_feat_st], dim=1)
        memory_2, prev2_feat_lt = self.forward_ltam(prev2_feat_st, prev_clips_enc)

        prev_clips_enc = torch.cat([prev3_feat_st, prev2_feat_lt, memory_2], dim=1)
        memory_1, prev1_feat_lt = self.forward_ltam(prev1_feat_st, prev_clips_enc)

        return prev2_feat_lt, prev1_feat_lt, memory_1

    def forward_curr_clips(self,
                           curr_images,
                           prev_feat):
        ndim = curr_images.ndim
        assert ndim == 5, \
            "Expected dimension of previous images should be 5, got {}".format(
                ndim)

        # current features with aggregated
        curr_feat = self.forward_backbone_attn(curr_images)
        curr_feat_x0 = curr_feat['dark5']
        if curr_feat_x0.ndim == 4:
            curr_feat_x0 = curr_feat_x0.unsqueeze(1)

        if isinstance(prev_feat, list) or isinstance(prev_feat, tuple):
            prev_feat = torch.cat(prev_feat, dim=1)

        if prev_feat is not None:
            if prev_feat.ndim == 5:
                b, t, c, h, w = prev_feat.shape
                # prev_feat = prev_feat.view(b, t * h * w, c)
            else:
                b, l, c = curr_feat['dark5'].shape
                h = w = int(math.sqrt(l))

            # if curr_feat_x0.ndim == 5:
            #     b, _, c, h, w = curr_feat_x0.shape
            #     curr_feat_x0 = curr_feat_x0.contiguous().view(b, h * w, c)
            _, curr_feat_x0 = self.forward_ltam(curr_feat_x0, prev_feat)
            # curr_x0 = curr_feat['dark5']
            curr_feat["dark5"] = curr_feat_x0.view(b, c, h, w)
        else:
            curr_feat['dark5'] = curr_feat["dark5"][0]
            neck_outputs = self.backbone.forward_pan(curr_feat)
            detections, detector_losses = self.box_head(neck_outputs, None)
            return curr_feat, detections

        neck_outputs = self.backbone.forward_pan(curr_feat)

        return neck_outputs

    def forward_infer(self, curr_images, memory):
        ndim = curr_images.ndim
        assert ndim == 5, \
            "Expected dimension of previous images should be 5, got {}".format(
                ndim)
        curr_feat = self.forward_backbone_attn(curr_images)
        # memory = torch.cat(memory, dim=2)

        # memory = memory.transpose(1, 2)
        curr_x0 = curr_feat['dark5']
        memory_long, x_lt = self.forward_ltam(curr_x0.contiguous().unsqueeze(0), memory)
        if x_lt.ndim == 5:
            x_lt = x_lt.flatten(0, 1)
        curr_feat["dark5"] = x_lt
        neck_outputs = self.backbone.forward_pan(curr_feat)

        return neck_outputs, x_lt, memory_long

    @staticmethod
    def _freeze_backbone_weight(backbone: torch.nn.Module):
        for i, (name, param) in enumerate(backbone.named_parameters()):
            if "resnet" in name:
                param.requires_grad = False


class SSDTMAPANMultiDetector(SSDTMAPANDetector):
    def __init__(self, cfg):
        super(SSDTMAPANMultiDetector, self).__init__(cfg)
        self.sequence_num = cfg.DATASETS.SEQUENCE_NUM
        self.memory_key_conv = nn.Sequential(
            SequenceConv(self.in_channels, self.key_channels, 1, 1,
                         self.conv_cfg, self.norm_cfg, self.act_cfg),
            SequenceConv(self.key_channels, self.key_channels, 3, 1,
                         self.conv_cfg, self.norm_cfg, self.act_cfg)
        )

        self.memory_value_conv = nn.Sequential(
            SequenceConv(self.in_channels, self.value_channels, 1, 1,
                         self.conv_cfg, self.norm_cfg, self.act_cfg),
            SequenceConv(self.value_channels, self.value_channels, 3, 1,
                         self.conv_cfg, self.norm_cfg, self.act_cfg)
        )
        self.memory_decoder = nn.Sequential(
            ConvModule(
                self.value_channels * 2,
                self.value_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                self.value_channels,
                self.value_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )

    def forward_memory_decoder(self, memory_readout):
        return self.memory_decoder(memory_readout)

    def forward(self, prev_images, curr_images, targets=None):
        # with torch.no_grad():
        #     prev_feat = self.forward_prev_clips(prev_images)
        prev_feat = self.forward_prev_clips(prev_images)
        torch.cuda.empty_cache()
        curr_agg_feat = self.forward_curr_clips(curr_images, prev_feat)

        if isinstance(curr_agg_feat, torch.Tensor):
            curr_agg_feat = [curr_agg_feat]

        detections, detector_losses = self.box_head(curr_agg_feat, targets)
        if self.training:
            return detector_losses
        return curr_agg_feat, detections

    def forward_prev_clips(self, prev_images: torch.Tensor):
        ndim = prev_images.ndim
        assert ndim == 6, \
            "Expected dimension of previous images should be 6, got {}".format(
                ndim)

        b, _, t = prev_images.shape[0:3]
        prev_images = torch.flatten(prev_images, start_dim=0, end_dim=1)
        # with torch.no_grad():
        #     prev_features = self.forward_backbone_and_neck(prev_images)
        prev_features = self.forward_backbone_stam(prev_images)

        prev_x0 = prev_features["dark5"]
        _, c, h, w = prev_x0.shape

        prev_x0 = prev_x0.view(self.sequence_num, b, c, h, w)
        prev_x0 = torch.chunk(prev_x0, chunks=self.sequence_num, dim=0)
        prev_x0_0, prev_x0_1 = prev_x0
        prev_mem_readout = self.forward_tma(
            prev_feat=prev_x0_0, curr_feat=prev_x0_1)

        return prev_mem_readout

    def forward_curr_clips(self,
                           curr_images,
                           prev_mem_readout):
        ndim = curr_images.ndim
        assert ndim == 5, \
            "Expected dimension of previous images should be 5, got {}".format(
                ndim)

        # with torch.no_grad():
        #     curr_feat = self.forward_backbone_and_neck(curr_images)

        # current features with aggregated
        curr_feat = self.forward_backbone_stam(curr_images)
        if prev_mem_readout is not None:
            curr_x0 = self.forward_tma(prev_mem_readout, curr_feat)
            curr_feat["dark5"] = curr_x0
        else:
            neck_outputs = self.backbone.forward_pan(curr_feat)
            detections, detector_losses = self.box_head(neck_outputs, None)
            return curr_feat, detections

        neck_outputs = self.backbone.forward_pan(curr_feat)
        return neck_outputs

    def forward_tma(self, prev_feat, curr_feat):
        """

        :param prev_feat: previous video clip features
        :param curr_feat: current video clip features
        :return: aggregated enhanced current video clip features
        """
        if isinstance(prev_feat, dict):
            prev_x0 = prev_feat["dark5"]
        else:
            prev_x0 = prev_feat

        if isinstance(curr_feat, dict):
            curr_x0 = curr_feat["dark5"]
        else:
            curr_x0 = curr_feat

        if prev_x0.ndim == 4:  # b(t) * c * h * w
            b, c, h, w = prev_x0.shape
            prev_x0 = prev_x0.view(1, -1, c, h, w)  # t * b * c * h * w

        if curr_x0.ndim == 5:
            curr_x0 = torch.squeeze(curr_x0, dim=0)

        prev_key, prev_value = self.encode_memory_kv(prev_x0)
        curr_key, curr_value = self.encode_query_kv(curr_x0)
        memory_readout = self.memory_module(
            prev_key, prev_value, curr_key, curr_value)
        memory_readout = self.bottleneck(memory_readout)

        return memory_readout


if __name__ == '__main__':
    from ssd.config import cfg

    cfg.merge_from_file(
        "../../../configs/resnet/dark/darknet_ssd320_dark_tma_pan.yaml")
    cfg.freeze()
    detector = SSDTMAPANMultiDetector(cfg)

    prev_tensors = torch.randn((2, 8, 3, 3, 320, 320))
    curr_tensors = torch.randn((8, 3, 3, 320, 320))
    detector(prev_tensors, curr_tensors)
