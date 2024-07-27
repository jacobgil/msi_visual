from argparse import Namespace
from pathlib import Path
import glob
import os


def display_path(path):
    try:
        metadata = eval(
            open(
                Path(path) /
                "args.txt").read().replace(
                "\\",
                "/"))
        bins = metadata.bins
        if 'id' in metadata:
            id = metadata.id + ' '
        else:
            id = ''
        return f"{id}m/z {metadata.start_mz}-{metadata.end_mz} {bins} bins"
    except Exception as e:
        print(e)
        return None


def display_paths_to_extraction_paths(extraction_root_folder):

    extraction_paths = glob.glob(
        str(Path(extraction_root_folder) / "**" / "args.txt"), recursive=True)
    extraction_paths = [Path(p).parent for p in extraction_paths]

    result = dict(zip([display_path(p)
                  for p in extraction_paths], extraction_paths))

    result = {k: v for k, v in result.items() if k is not None}
    return result


def get_files_from_folder(path):
    path = Path(path)
    return glob.glob(os.path.join(path, "*.npy"))
