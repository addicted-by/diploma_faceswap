import typing as t
import abc
from omegaconf import DictConfig
import cv2
from pathlib import Path
from pprint import pprint
import numpy as np
from utils import faceshifter_swap_faces


class BaseInferencer(t.Protocol):
    cfg: DictConfig

    @abc.abstractmethod
    def preprocess(self):
        ...

    @abc.abstractmethod
    def prepare_folders(self):
        ...

    @abc.abstractmethod
    def inference(self):
        ...

    @abc.abstractmethod
    def save_result(self):
        ...

    @abc.abstractproperty
    def get_source(self) -> np.ndarray:
        ...

    @abc.abstractproperty
    def get_target(self) -> np.ndarray:
        ...

    def check_validity(self):
        source_path = Path(self.cfg.source)
        target_path = Path(self.cfg.target)
        assert (
            source_path.exists()
            and
            target_path.exists()
        ), "Check the proposed paths!"

    def __call__(self, args, kwargs):
        self.inference(args, kwargs)

    def __repr__(self) -> str:
        return self.__class__.__name__


class FaceShifterInferencer(BaseInferencer):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        pprint(self)
        pprint(self.cfg, sort_dicts=False)

    @property
    def get_source(self) -> np.ndarray:
        return self.source

    @property
    def get_target(self) -> np.ndarray:
        return self.target

    def preprocess(self):
        self.check_validity()
        self.prepare_folders()
        self.source = cv2.imread(self.cfg.source)
        self.target = cv2.imread(self.cfg.target)


    def prepare_folders(self):
        pass

    def inference(self):
        self.preprocess()
        self.result = faceshifter_swap_faces(self.source, self.target)
        self.save_result()
