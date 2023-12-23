from abc import ABC, abstractmethod

class Policy(ABC):
    @abstractmethod
    def select_action(self, obs):
        pass

    @abstractmethod
    def get_memory_associations(self):
        pass
