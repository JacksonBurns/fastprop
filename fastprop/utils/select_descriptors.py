from mordred import descriptors, get_descriptors_in_module
from importlib.metadata import version
from packaging.version import Version


def mordred_descriptors_from_strings(string_list):
    out = []
    __version__ = Version(version("mordredcommunity"))
    for mdl in descriptors.all:
        for Desc in get_descriptors_in_module(mdl, submodule=False):
            for desc in Desc.preset(__version__):
                if desc.__str__() in string_list:
                    out.append(desc)
    return out
