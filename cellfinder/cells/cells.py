import os
import re
import math
import logging

from functools import total_ordering

from xml.etree import ElementTree
from xml.etree.ElementTree import Element as EtElement


@total_ordering
class Cell(object):
    ARTIFACT = -1
    CELL = 2
    UNKNOWN = 1

    # for classification compatibility
    NO_CELL = 1

    def __init__(self, pos, cell_type):
        if isinstance(pos, str):
            pos = pos_from_file_name(os.path.basename(pos))
        if isinstance(pos, ElementTree.Element):
            pos = pos_from_xml_marker(pos)
        if isinstance(pos, dict):
            pos = pos_from_dict(pos)
        pos = self._sanitize_position(pos)
        x, y, z = [int(p) for p in pos]
        self.x = x
        self.y = y
        self.z = z

        self.transformed_x, self.transformed_y, self.transformed_z = x, y, z

        self.structure_id = None
        self.hemisphere = None

        if cell_type is None:
            self.type = Cell.UNKNOWN
        elif str(cell_type).lower() == "cell":
            self.type = Cell.CELL
        elif str(cell_type).lower() == "no_cell":
            self.type = Cell.ARTIFACT
        else:
            self.type = int(cell_type)

    def _sanitize_position(self, pos, verbose=True):
        out = []
        for coord in pos:
            if math.isnan(coord):
                if verbose:
                    print(
                        "WARNING: NaN position for for cell\n"
                        "defaulting to 1"
                    )
                coord = 1
            out.append(coord)
        return out

    def _transform(
        self,
        x_scale=1.0,
        y_scale=1.0,
        z_scale=1.0,
        x_offset=0,
        y_offset=0,
        z_offset=0,
        integer=False,
    ):
        x = self.x
        y = self.y
        z = self.z

        x += x_offset
        y += y_offset
        z += z_offset

        x *= x_scale
        y *= y_scale
        z *= z_scale

        if integer:
            return [int(round(e)) for e in (x, y, z)]
        else:
            return x, y, z

    def transform(
        self,
        x_scale=1.0,
        y_scale=1.0,
        z_scale=1.0,
        x_offset=0,
        y_offset=0,
        z_offset=0,
        integer=False,
    ):
        transformed_coords = self._transform(
            x_scale, y_scale, z_scale, x_offset, y_offset, z_offset, integer
        )
        self.x, self.y, self.z = transformed_coords

    def soft_transform(
        self,
        x_scale=1.0,
        y_scale=1.0,
        z_scale=1.0,
        x_offset=0,
        y_offset=0,
        z_offset=0,
        integer=False,
    ):
        transformed_coords = self._transform(
            x_scale, y_scale, z_scale, x_offset, y_offset, z_offset, integer
        )
        (
            self.transformed_x,
            self.transformed_y,
            self.transformed_z,
        ) = transformed_coords

    def flip_x_y(self):
        self.y, self.x = self.x, self.y

    def is_cell(self):
        return self.type == Cell.CELL

    def to_xml_element(self):
        sub_elements = [EtElement("Marker{}".format(axis)) for axis in "XYZ"]
        coords = [int(coord) for coord in (self.x, self.y, self.z)]
        for sub_element, coord in zip(sub_elements, coords):
            if coord < 1:
                print(
                    "WARNING: negative coordinate found at {}\n"
                    "defaulting to 1".format(coord)
                )
                coord = 1  # FIXME:
            sub_element.text = str(coord)

        element = EtElement("Marker")
        element.extend(sub_elements)
        return element

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (self.x, self.y, self.z, self.type) == (
            other.x,
            other.y,
            other.z,
            other.type,
        )

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        if self == other:
            return False
        try:
            if self.z < other.z:
                return True
            elif self.z > other.z:
                return False
            elif self.y < other.y:
                return True
            elif self.y > other.y:
                return False
            elif self.x < other.x:
                return True
            else:
                return False
        except AttributeError as err:
            return NotImplementedError(
                "comparison to {} is not implemented, {}".format(
                    type(other), err
                )
            )

    def __str__(self):
        return "Cell: x: {}, y: {}, z: {}, type: {}".format(
            int(self.x), int(self.y), int(self.z), self.type
        )

    def __repr__(self):
        return "{}, ({}, {})".format(
            self.__class__, [self.x, self.y, self.z], self.type
        )

    def to_dict(self):
        return {"x": self.x, "y": self.y, "z": self.z, "type": self.type}

    def __hash__(self):
        return hash(str(self))


class UntypedCell(Cell):
    def __init__(self, pos):
        super(UntypedCell, self).__init__(pos, self.UNKNOWN)

    @property
    def type(self):
        return self.UNKNOWN

    @type.setter
    def type(self, value):
        pass

    @classmethod
    def from_cell(cls, cell):
        return cls([cell.x, cell.y, cell.z])

    def to_cell(self):
        return Cell([self.x, self.y, self.z], self.type)


def pos_from_dict(position_dict):
    return [position_dict["x"], position_dict["y"], position_dict["z"]]


def pos_from_xml_marker(element):
    marker_names = ["Marker{}".format(axis) for axis in "XYZ"]
    pos = [element.find(marker_name).text for marker_name in marker_names]
    return [float(num) for num in pos]


def pos_from_file_name(file_name):
    x = re.findall(r"x\d+", file_name.lower())
    y = re.findall(r"y\d+", file_name.lower())
    z = re.findall(r"z\d+", file_name.lower())
    return [int(p) for p in (x[-1][1:], y[-1][1:], z[-1][1:])]


def transform(cell, deformation_field, field_scales, scales):
    """
    Transforms cell position from one space, to another (defined by a
    deformation field)
    :param cell: Cells in original space
    :param deformation_field: Deformation field
    (shape (len(x), len(y), len(z), 3). For each spatial position, there is a
    vector mapping onto a new coordinate space.
    :param field_scales: Scaling of the deformation field values (in mm) into
    voxel space (e.g. 100,100,100)
    :param scales: Scale of cell x, y and z positions onto deformation
    field (e.g. 0.2, 0.2, 0.5)
    :return: Cell in the new space
    """
    scaled_x = int(round(cell.x * scales[0]))
    scaled_y = int(round(cell.y * scales[1]))
    scaled_z = int(round(cell.z * scales[2]))

    try:
        new_x = int(
            round(
                field_scales[0]
                * deformation_field[scaled_x, scaled_y, scaled_z, 0, 0]
            )
        )
        new_y = int(
            round(
                field_scales[1]
                * deformation_field[scaled_x, scaled_y, scaled_z, 0, 1]
            )
        )
        new_z = int(
            round(
                field_scales[2]
                * deformation_field[scaled_x, scaled_y, scaled_z, 0, 2]
            )
        )

        # if any new coordinates are negative
        if any(position < 0 for position in [new_x, new_y, new_z]):
            warn_outside_target_space(cell)

        else:
            cell.x = new_x
            cell.y = new_y
            cell.z = new_z
        return cell

    except IndexError:
        warn_outside_target_space(cell)


def warn_outside_target_space(cell):
    logging.warning(
        "Position x:{}, y:{}, z{} is outside the target "
        "coordinate space, skipping. If this happens for many "
        "cells, something may be up.".format(cell.x, cell.y, cell.z)
    )


def transform_cell_positions(
    cells, deformation_field, field_scales=(100, 100, 100), scales=(1, 1, 1)
):
    """
    Transforms cell positions from one space, to another (defined by a
    deformation field)
    :param cells: List of cells in original space
    :param deformation_field: Deformation field
    (shape (len(x), len(y), len(z), 3). For each spatial position, there is a
    vector mapping onto a new coordinate space.
    :param field_scales: Scaling of the deformation field values (in mm) into
    voxel space (e.g. 100,100,100)
    :param scales: Scale of cell x, y and z positions onto deformation
    field (e.g. 0.2, 0.2, 0.5)
    :return: list of cells in the new space
    """
    # TODO: parallelise (maybe not needed, very quick anyway)
    # TODO: clarify this transformation, and the existing transformed_x
    # property of the cells used for other things (e.g. summaries)
    transformed_cells = [
        transform(cell, deformation_field, field_scales, scales)
        for cell in cells
    ]

    # Remove None's from list (where cell couldn't be transformed)
    transformed_cells = [cell for cell in transformed_cells if cell]
    cells_not_transformed = len(cells) - len(transformed_cells)
    logging.warning(
        "{} cells were not transformed to standard space".format(
            cells_not_transformed
        )
    )
    return transformed_cells
