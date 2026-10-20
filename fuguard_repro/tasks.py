import torch
from .models import ConvNet, LSTMClassifier
from .proxy import image_proxy, awe_proxy, load_vae


class BaseTask:
    """What the FL engine needs from a dataset: per-client tensors, test sets, a model factory, a proxy maker."""
    kind = "base"

    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device
        self._clients = {}
        self.clean_test = self.poison_test = None
        self._vae = None

    @property
    def client_names(self):
        return list(self._clients.keys())

    def client_data(self, name):
        return self._clients[name]

    @property
    def target_label(self):
        return self.cfg["target_label"]

    def make_model(self):
        raise NotImplementedError

    def make_proxy(self, x, y, cfg):
        raise NotImplementedError


class ImageTask(BaseTask):
    kind = "image"

    def __init__(self, cfg, device, clients, clean_test, poison_test):
        super().__init__(cfg, device)
        self._clients = {k: (x.to(device), y.to(device)) for k, (x, y) in clients.items()}
        self.clean_test = tuple(t.to(device) for t in clean_test)
        self.poison_test = tuple(t.to(device) for t in poison_test)

    def make_model(self):
        return ConvNet(im_size=tuple(self.cfg["image_size"]), num_classes=self.cfg["num_classes"]).to(self.device)

    def make_proxy(self, x, y, cfg):
        if self._vae is None:
            self._vae = load_vae(cfg["vae_id"], self.device)
        return image_proxy(self._vae, x, y, cfg, self.device)


class AWETask(BaseTask):
    kind = "awe"

    def __init__(self, cfg, device, clients, clean_test, poison_test):
        super().__init__(cfg, device)
        self._clients = {k: (x.to(device), y.to(device)) for k, (x, y) in clients.items()}
        self.clean_test = tuple(t.to(device) for t in clean_test)
        self.poison_test = tuple(t.to(device) for t in poison_test)

    def make_model(self):
        return LSTMClassifier(input_dim=self.cfg["input_dim"], num_classes=self.cfg["num_classes"]).to(self.device)

    def make_proxy(self, x, y, cfg):
        return awe_proxy(x, y, cfg, self.device)
