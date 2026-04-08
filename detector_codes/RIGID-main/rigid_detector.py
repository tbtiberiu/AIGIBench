import torch
import torch.nn.functional as F

class RIGID_Detector:
    def __init__(self, lamb=0.05):
        self.lamb = lamb
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        self.model.eval()

    @torch.no_grad()
    def calculate_sim(self, data):
        features = self.model(data)
        noise = torch.randn_like(data).to(data.device)
        trans_data = data + noise * self.lamb
        trans_features = self.model(trans_data)
        sim_feat = F.cosine_similarity(features, trans_features, dim=-1)
        return sim_feat

    @torch.no_grad()
    def detect(self, data):
        sim = self.calculate_sim(data)
        # Higher similarity means more "real" (stable under noise)
        # We return 1 - sim so that high = fake, to align with evaluate_detectors.py
        return 1.0 - sim
