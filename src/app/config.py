"""Application configuration settings"""
from pydantic_settings import BaseSettings
from pathlib import Path
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Database settings
    database_url: str
    debug: bool = False

    # Imputation results CSV storage path
    impute_results_path: str = "data/imputation_results"

    class Config:
        env_file = ".env" if os.getenv("ENV") != "cluster" else None
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def impute_results_dir(self) -> Path:
        """Get the imputation results directory as a Path object"""
        return Path(self.impute_results_path)


# Global settings instance
settings = Settings()
