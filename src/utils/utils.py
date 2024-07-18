import json
import io

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def jsave(f, dicts, mode="w", indent=4):
    """Save a dictionary into a .json file"""
    f = _make_r_io_base(f, mode)
    json.dump(dicts, f, indent=indent)
    f.close()


    
    