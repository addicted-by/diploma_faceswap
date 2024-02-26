import hydra
from omegaconf import OmegaConf, DictConfig
import inferencers



@hydra.main(
        version_base="1.3",
        config_path=".",
        config_name="inference.yaml"
)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    inferencer = getattr(inferencers, cfg.inferencer.name)(cfg.inferencer)
    inferencer()

if __name__ == "__main__":
    main()
