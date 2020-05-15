from abc import ABC, abstractmethod


class ModelSearcherABC(ABC):
    @abstractmethod
    def compute_grid_search(self):
        pass
