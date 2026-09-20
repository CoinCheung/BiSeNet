#!/usr/bin/python
# -*- encoding: utf-8 -*-

from lib.data.rugd4_dataset import RUGD4Dataset


class RUGD3Dataset(RUGD4Dataset):
    """
    RUGD three-class traversability dataset.

    Class IDs:
        0: sky
        1: traversable
        2: non-traversable
      255: ignore
    """

    pass
