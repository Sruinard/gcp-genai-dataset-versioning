import pathlib
import dataclasses
import yaml
import enum

@dataclasses.dataclass
class Config:
    project_id: str
    bucket_name: str
    location: str

class Mode(enum.Enum):
    DEV = "dev"
    PROD = "prod"
    

def new(mode: Mode = Mode.DEV):
    config_path = pathlib.Path(f'../assets/configs/{mode.value}.yaml')
    if not config_path.exists():
        raise FileNotFoundError(f"Config file for mode {mode.value} does not exist.")
    with config_path.open() as f:
        config_data = yaml.safe_load(f)
        return Config(**config_data)
