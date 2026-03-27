SR_REGISTRY = {}
FUSION_REGISTRY = {}

def register_sr(name):
    def deco(cls):
        SR_REGISTRY[name] = cls
        return cls
    return deco

def register_fusion(name):
    def deco(cls):
        FUSION_REGISTRY[name] = cls
        return cls
    return deco

def build_sr(name, **kwargs):
    if name not in SR_REGISTRY:
        raise KeyError(f"Unknown SR model: {name}")
    return SR_REGISTRY[name](**kwargs)

def build_fusion(name, **kwargs):
    if name not in FUSION_REGISTRY:
        raise KeyError(f"Unknown fusion module: {name}")
    return FUSION_REGISTRY[name](**kwargs)
