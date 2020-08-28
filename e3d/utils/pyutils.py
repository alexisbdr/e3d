from copy import deepcopy

def from_dict(cls, dict_in):
    mod = deepcopy(dict_in)
    use_dict = {}
    for entry in mod:
        if entry in cls.__annotations__:
            use_dict[entry] = mod[entry]
    return cls(**use_dict)
