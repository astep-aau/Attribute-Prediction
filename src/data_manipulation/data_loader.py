from abc import ABC, abstractmethod
import pandas as pd

class DataLoader(ABC):
    @abstractmethod
    def get_travel_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_meta_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_adjacency(self) -> pd.DataFrame:
        pass
