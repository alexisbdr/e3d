
def from_dict(cls, dict_in):
    mod = dict_in.copy()
    use_dict = {}
    for entry in mod:
        if entry in cls.__annotations__:
            use_dict[entry] = dict_in[entry]
    return cls(**use_dict)
