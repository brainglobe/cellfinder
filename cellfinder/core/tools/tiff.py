from os import listdir
from os.path import isfile, join

import natsort
from brainglobe_utils.cells.cells import Cell, UntypedCell


class TiffList:
    """
    Represents a list of tiff files output from the cell extractor plugin and
    used for training and classification.

    Represents tiff files whose names end with `Ch[ch].tif`, where `[ch]` is
    the non-zero-padded channel number, starting with channel 1. Given a list
    of tiff files for the first channel (`...Ch1.tif`), it will find the other
    corresponding tif files for the channels passed in the `channels`
    parameter.

    :param ch1_list: List of the paths to the tiff files of the first channel.
    :param channels: List of channels numbers for which we have tif files
        (zero or one based).
    :param label: Label of all these tiffs (e.g. `"cell"`, `"no_cell"`).
    """

    def __init__(
        self,
        ch1_list: list[str],
        channels: list[int],
        label: str | None = None,
    ):
        self.ch1_list = natsort.natsorted(ch1_list)
        self.label = label
        self.channels = channels

    def make_tifffile_list(self) -> list["TiffFile"]:
        """
        Splits the list of tiffs represented by this instance into a list of
        `TiffFile` instances.

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
    """Like `TiffList` except it takes a directory (`tiff_dir`) and gets all
    the tiffs in that directory that match the `Ch[ch].tif` pattern.
    """

    def __init__(
        self,
        tiff_dir: str,
        channels: list[int],
        label: str | None = None,
    ):
        super(TiffDir, self).__init__(
            [
                join(tiff_dir, f)
                for f in listdir(tiff_dir)
                if f.lower().endswith("ch" + str(channels[0]) + ".tif")
            ],
            channels,
            label,
        )


class TiffFile:
    """This class represents a multichannel tiff file, with one individual
    file per channel.

    :param path: The full path to the first (zero or one based) channel's tiff
        file.
    :param channels: List of channels numbers for which we have tiff files
        next to the first channel. It must match the `Ch[ch].tif` pattern.
    :param label: Label of the tiffs (e.g. `"cell"`, `"no_cell"`).
    """

    def __init__(
        self,
        path: str,
        channels: list[int],
        label: str | None = None,
    ):
        self.path = path
        self.channels = channels
        self.label = label

    def files_exist(self):
        """
        Returns whether the tiffs actually exist on disk for all the channels
        represented by this instance.
        """
        return all([isfile(tif) for tif in self.img_files])

    def as_cell(self, force_typed=True) -> Cell | UntypedCell:
        """
        Returns a `Cell` instance that represents the (potential) cell for
        whom the tiff files was saved.

        :param force_typed: If True returns a `Cell`. If False, it returns a
            `UntypedCell` if `self.label` is None and `Cell` otherwise.
        :return:
        """
        if force_typed:
            match self.label:
                case None:
                    cell_type = Cell.ARTIFACT
                case "cell":
                    cell_type = Cell.CELL
                case "no_cell":
                    cell_type = Cell.NO_CELL
                case _:
                    raise ValueError(f"Unknown cell type {self.label}")

            return Cell(self.path, cell_type)

        if self.label is None:
            return UntypedCell(self.path)
        return Cell(self.path, self.label)

    @property
    def img_files(self) -> list[str]:
        """
        Returns a list of the full filenames to the tiffs for all channels
        represented by this instance.
        """
        return [self.path[:-5] + str(ch) + ".tif" for ch in self.channels]
