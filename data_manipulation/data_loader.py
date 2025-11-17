from abc import ABC, abstractmethod
import pandas as pd

class DataLoader(ABC): 
    @abstractmethod
    def getTravelData(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def getMetaData(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def getAdjacency(self) -> pd.DataFrame:
        pass
