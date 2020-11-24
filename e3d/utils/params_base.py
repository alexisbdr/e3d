from dataclasses import asdict, dataclass

from utils.pyutils import _from_dict


@dataclass
class ParamsBase:
    def __post_init__(self):
        if self.config_file:
            with open(self.config_file) as f:
                dict_in = json.load(f)
                for key, value in dict_in.items():
                    if key in self.__annotations__:
                        setattr(self, key, value)

    @classmethod
    def from_dict(cls, dict_in: dict):
        return _from_dict(cls, dict_in)
