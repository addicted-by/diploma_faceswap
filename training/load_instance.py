from huggingface_hub import snapshot_download
import hydra
from omegaconf import OmegaConf, DictConfig

@hydra.main(
        version_base="1.3",
        config_path="./load_cfg",
        config_name="instance.yaml"
)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    snapshot_download(**cfg.snapshot)


if __name__ == "__main__":
    main()
