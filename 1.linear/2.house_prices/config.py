from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from enum import StrEnum

class Devices(StrEnum):
    cpu = 'cpu'
    mps = 'mps'
    cuda = 'cuda'


class Settings(BaseSettings):
    seed: int = Field(..., gt=0)
    device: Devices
    epochs: int = Field(..., gt=0)
    batch_size: int = Field(..., gt=0)

    model_config = SettingsConfigDict(env_file='.env')

settings = Settings()