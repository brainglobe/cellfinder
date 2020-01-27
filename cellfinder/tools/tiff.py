import natsort

from os import listdir
from os.path import isfile, join

from imlib.cells.cells import Cell, UntypedCell


class TiffList(object):
    """This class represents list of tiff files. These tiff files are the
     output from the cell extractor plugin
    and are used as training and classification data.
    """

    def __init__(self, ch1_list, channels, label=None):
        """ A list of tiff files output by the cell extractor plugin to be
        used in machine learning.
        Expects file names to end with "ch[ch].tif", where [ch] is the
        non-zero-padded channel index.
        Given a list of tiff files for the first channel, it will find the
        corresponding files for the
        channels passed in the [channels] parameter.
        :param ch1_list: List of the tiff files of the first channel.
        :param channels: List of channels to use.
        :param label: Label of the directory (e.g. 2 for cell, 1 for no cell).
        Can be ignored on classification runs.
        """

        self.ch1_list = natsort.natsorted(ch1_list)
        self.label = label
        self.channels = channels

    def make_tifffile_list(self):
        """

        :return: Returns the relevant tiff files as a list of TiffFile objects.
        """
        files = [
            f
            for f in self.ch1_list
            if f.lower().endswith("ch" + str(self.channels[0]) + ".tif")
        ]

        tiff_files = [
            TiffFile(tiffFile, self.channels, self.label) for tiffFile in files
        ]
        return tiff_files


class TiffDir(TiffList):
    """A simplified version of TiffList that uses all tiff files without
    any filtering.
    """

    def __init__(self, tiff_dir, channels, label=None):
        super(TiffDir, self).__init__(
            [
                join(tiff_dir, f)
                for f in listdir(tiff_dir)
                if f.lower().endswith("ch" + str(channels[0]) + ".tif")
            ],
            channels,
            label,
        )


class TiffFile(object):
    """This class represents a multichannel tiff file, with one individual
    file per channel.
    """

    def __init__(self, path, channels, label=None):
        self.path = path
        self.channels = channels
        self.label = label

    def files_exist(self):
        return all([isfile(tif) for tif in self.img_files])

    def as_cell(self, force_typed=True):
        if force_typed:
            return (
                Cell(self.path, -1)
                if self.label is None
                else Cell(self.path, self.label)
            )
        else:
            return (
                UntypedCell(self.path)
                if self.label is None
                else Cell(self.path, self.label)
            )

    @property
    def img_files(self):
        return [self.path[:-5] + str(ch) + ".tif" for ch in self.channels]
