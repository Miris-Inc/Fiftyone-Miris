"""Export a Miris dataset you built in the FiftyOne App into this folder as a
``FiftyOneDataset`` snapshot, so it ships as a remotely-sourced zoo dataset.

Usage::

    python dataset/export_snapshot.py <dataset_name>

This writes ``metadata.json``, ``samples.json``, ``data/`` (the ``.fo3d``
scenes), and ``fields/`` (the ``miris_opm`` thumbnails) next to this script.
All media is copied in and paths are stored relative, so the snapshot is
fully self-contained and portable. After exporting, commit the new files and
optionally update ``size_samples`` in ``fiftyone.yml``.
"""
import argparse
import os

import fiftyone as fo

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_name", help="name of the FiftyOne dataset to snapshot"
    )
    parser.add_argument(
        "--export-dir",
        default=HERE,
        help="directory to export into (default: this folder)",
    )
    args = parser.parse_args()

    dataset = fo.load_dataset(args.dataset_name)
    dataset.export(
        export_dir=args.export_dir,
        dataset_type=fo.types.FiftyOneDataset,
        export_media=True,
    )
    print(f"Exported {len(dataset)} samples to {args.export_dir}")
    print("Remember to update `size_samples` in fiftyone.yml and commit the snapshot.")


if __name__ == "__main__":
    main()
