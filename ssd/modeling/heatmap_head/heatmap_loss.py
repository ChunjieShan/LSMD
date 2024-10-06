import torch


class KpLoss(object):
    def __init__(self):
        self.criterion = torch.nn.MSELoss(reduction='none')

    def __call__(self, logits, targets):
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        bs = logits.shape[0]
        np_logits = logits.detach().cpu().squeeze().numpy()
        # [num_kps, H, W] -> [B, num_kps, H, W]
        # heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])
        # for i in range(np_logits.shape[0]):
        #     plt.figure()
        #     plt.imshow(np_logits[i])
        #     plt.show()
        heatmaps = targets["heatmap"].to(device, dtype=torch.float32)
        np_heatmaps = heatmaps.detach().cpu().squeeze().numpy()

        # =========== Plotting for debugging ================
        # for i in range(np_logits.shape[0]):
        #     plt.figure()
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(np_logits[i])
        #     plt.title("Prediction")
        #
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(np_heatmaps[i])
        #     plt.title("GT")
        #
        #     plt.show()
        # [num_kps] -> [B, num_kps]
        # kps_weights = torch.stack([t["kps_weights"].to(device) for t in targets])
        kps_weights = targets["kps_weights"].to(device)

        # [B, num_kps, H, W] -> [B, num_kps]
        loss = self.criterion(logits, heatmaps).mean(dim=[2, 3])
        loss = torch.sum(loss * kps_weights) / bs
        return loss
