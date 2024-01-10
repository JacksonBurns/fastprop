from importlib.metadata import version

from mordred import descriptors, get_descriptors_in_module
from packaging.version import Version


def mordred_descriptors_from_strings(string_list, include_3d=False):
    """
    returns mordred descriptor classes based on their string name
    if string_list is empty, returns all
    include_3d will include or exclude descriptors that require 3d info
    """
    out = []
    string_set = set(string_list)
    if (input_length := len(string_list)) != (set_length := len(string_set)):
        raise RuntimeError(f"Input contains {input_length - set_length} duplicate entires.")
    __version__ = Version(version("mordredcommunity"))
    for mdl in descriptors.all:
        for Desc in get_descriptors_in_module(mdl, submodule=False):
            for desc in Desc.preset(__version__):
                if not string_list or desc.__str__() in string_list:
                    if desc.require_3D and not include_3d:
                        continue
                    out.append(desc)
    return out
