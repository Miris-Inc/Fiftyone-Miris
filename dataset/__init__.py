"""Miris demo dataset — remotely-sourced FiftyOne zoo dataset.

This dataset is a snapshot of Miris 3D assets, exported via FiftyOne's
``FiftyOneDataset`` format and committed alongside this module. Each sample's
``filepath`` is a ``.fo3d`` scene referencing a first-class ``MirisStream``
node (rendered natively by FiftyOne core's looker-3d), and ``miris_opm``
carries an ``OrthographicProjectionMetadata`` thumbnail so the grid renders a
2D preview.

The snapshot files (``metadata.json``, ``samples.json``, ``data/``,
``fields/``) live in this directory and are copied into the local zoo
directory when the dataset is loaded, so there is nothing to download from a
remote host.

Load it with::

    import fiftyone.zoo as foz

    dataset = foz.load_zoo_dataset("Miris-Inc/Fiftyone-Miris")
"""
import json
import os

import fiftyone as fo


def download_and_prepare(dataset_dir, split=None, **kwargs):
    """Prepares the in-repo Miris snapshot for import.

    The snapshot media and manifest are shipped in this directory and have
    already been copied into ``dataset_dir`` by FiftyOne, so there is nothing
    to download. We report the on-disk format (``FiftyOneDataset``) and the
    sample count; FiftyOne handles the import from there.

    Args:
        dataset_dir: the directory in which the dataset is constructed
        split (None): unused; this dataset has no splits
        **kwargs: unused

    Returns:
        a tuple of ``(fiftyone.types.FiftyOneDataset, num_samples, None)``
    """
    num_samples = _count_samples(dataset_dir)
    return fo.types.FiftyOneDataset, num_samples, None


def _count_samples(dataset_dir):
    """Counts samples in a FiftyOneDataset export (single-file or sharded)."""
    samples_json = os.path.join(dataset_dir, "samples.json")
    if os.path.isfile(samples_json):
        with open(samples_json) as f:
            return len(json.load(f).get("samples", []))

    # Large exports may shard samples into a `samples/` directory
    samples_dir = os.path.join(dataset_dir, "samples")
    if os.path.isdir(samples_dir):
        return sum(1 for n in os.listdir(samples_dir) if n.endswith(".json"))

    return 0
