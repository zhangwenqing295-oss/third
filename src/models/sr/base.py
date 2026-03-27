import torch.nn as nn

class BaseSRModel(nn.Module):
    def extract_features(self, x, stage="deep"):
        raise NotImplementedError
