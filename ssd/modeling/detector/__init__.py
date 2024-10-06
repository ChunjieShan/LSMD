from .ssd_detector import SSDDetector, SSDXMemDetector, SSDTMADetector, SSDTMAPANDetector, SSDTMAPANMultiDetector, SSDLSTPANDetector

_DETECTION_META_ARCHITECTURES = {
    "SSDDetector": SSDDetector,
    "SSDXMemDetector": SSDXMemDetector,
    "SSDTMADetector": SSDTMADetector,
    "SSDTMAPANDetector": SSDTMAPANDetector,
    "SSDLSTPANDetector": SSDLSTPANDetector,
    "SSDTMAPANMultiDetector": SSDTMAPANMultiDetector
}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
