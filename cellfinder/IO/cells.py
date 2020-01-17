import logging
import os
import yaml

import pandas as pd

from xml.dom import minidom
from xml.etree import ElementTree
from xml.etree.ElementTree import Element as EtElement

from cellfinder.cells.cells import Cell, UntypedCell, pos_from_file_name
from cellfinder.cells.tools import MissingCellsError
from cellfinder.tools.system import replace_extension


def get_cells(cells_file_path, cells_only=False, cell_type=None):
    # TODO: implement csv read
    if cells_file_path.endswith(".xml"):
        return get_cells_xml(cells_file_path, cells_only=cells_only)
    elif cells_file_path.endswith(".yml"):
        # Not general
        return get_cells_yml(cells_file_path, ignore_type=True)
    elif os.path.isdir(cells_file_path):
        try:
            return get_cells_dir(cells_file_path, cell_type=cell_type)
        except IndexError:
            # if a directory is given, but it contains
            # files that can't be read. Usually if the user gives the wrong
            # directory as input to `cellfinder_gen_cubes`
            raise_cell_read_error(cells_file_path)
    else:
        raise_cell_read_error(cells_file_path)


def raise_cell_read_error(cells_file_path):
    logging.error(
        "File format of: {} is not supported or contains errors. Please "
        "supply an xml file, or a directory of files with positions in the "
        "filenames."
        "".format(cells_file_path)
    )
    raise NotImplementedError(
        "File format of: {} is not supported or contains errors. Please "
        "supply an xml file, or a directory of files with positions in the "
        "filenames."
        "".format(cells_file_path)
    )


def get_cells_xml(xml_file_path, cells_only=False):
    with open(xml_file_path, "r") as xml_file:
        root = ElementTree.parse(xml_file).getroot()
        cells = []
        for type_marker in root.find("Marker_Data").findall("Marker_Type"):
            cell_type = int(type_marker.find("Type").text)
            for cell_element in type_marker.findall("Marker"):
                cells.append(Cell(cell_element, cell_type))
        if not cells:
            raise MissingCellsError(
                "No cells found in file {}".format(xml_file_path)
            )
    if cells_only:
        cells = [c for c in cells if c.is_cell()]
    return cells


def get_cells_yml(cells_file_path, ignore_type=False, marker="markers"):
    if not ignore_type:
        raise NotImplementedError(
            "Parsing cell types is not yet implemented for YAML files. "
            "Currently the only option is to merge them. Please try again with"
            " 'ignore_type=True'."
        )
    else:
        with open(cells_file_path, "r") as yml_file:
            data = yaml.safe_load(yml_file)
        cells = []
        for cell_type in list(data.keys()):
            type_dict = data[cell_type]
            if marker in type_dict.keys():
                for cell in type_dict[marker]:
                    cells.append(Cell(cell, Cell.UNKNOWN))
    return cells


def get_cells_dir(cells_file_path, cell_type=None):
    cells = []
    for file in os.listdir(cells_file_path):
        # ignore hidden files
        if not file.startswith("."):
            cells.append(Cell(file, cell_type))
    return cells


def save_cells(
    cells,
    xml_file_path,
    save_csv=False,
    indentation_str="  ",
    artifact_keep=True,
):
    # Assume always save xml file, and maybe save other formats
    cells_to_xml(
        cells,
        xml_file_path,
        indentation_str=indentation_str,
        artifact_keep=artifact_keep,
    )

    if save_csv:
        csv_file_path = replace_extension(xml_file_path, "csv")
        cells_to_csv(cells, csv_file_path)


def cells_to_xml(
    cells, xml_file_path, indentation_str="  ", artifact_keep=True
):
    xml_data = make_xml(cells, indentation_str, artifact_keep=artifact_keep)
    with open(xml_file_path, "w") as xml_file:
        xml_file.write(str(xml_data, "UTF-8"))


def cells_xml_to_df(xml_file_path):
    cells = get_cells(xml_file_path)
    return cells_to_dataframe(cells)


def cells_to_dataframe(cells):
    return pd.DataFrame([c.to_dict() for c in cells])


def cells_to_csv(cells, csv_file_path):
    df = cells_to_dataframe(cells)
    df.to_csv(csv_file_path)


def make_xml(cell_list, indentration_str, artifact_keep=True):
    root = EtElement("CellCounter_Marker_File")
    image_properties = EtElement("Image_Properties")
    file_name = EtElement("Image_Filename")
    file_name.text = "placeholder.tif"
    image_properties.append(file_name)
    root.append(image_properties)

    marker_data = EtElement("Marker_Data")
    current_type = EtElement("Current_Type")
    current_type.text = str(1)  # TODO: check
    marker_data.append(current_type)

    cell_dict = make_type_dict(cell_list)

    # if artifacts exist, do something with them (convert/delete)
    if Cell.ARTIFACT in cell_dict:
        cell_dict = deal_with_artifacts(cell_dict, artifact_keep=artifact_keep)

    for cell_type, cells in cell_dict.items():
        type_el = EtElement("Type")
        type_el.text = str(cell_type)
        mt = EtElement("Marker_Type")
        mt.append(type_el)
        for cell in cells:
            mt.append(cell.to_xml_element())
        marker_data.append(mt)
    root.append(marker_data)

    return pretty_xml(root, indentration_str)


def deal_with_artifacts(cell_dict, artifact_keep=True):
    # What do we want to do with the artifacts (if they exist)?
    # Might want to keep them for training
    if artifact_keep:
        logging.debug("Keeping artifacts")
        # Change their type
        for idx, artifact in enumerate(cell_dict[Cell.ARTIFACT]):
            cell_dict[Cell.ARTIFACT][idx].type = Cell.UNKNOWN
        # Add them to "cell_type = UNKNOWN" list
        cell_dict[Cell.UNKNOWN].extend(cell_dict[Cell.ARTIFACT])
    else:
        logging.debug("Removing artifacts")
    del cell_dict[Cell.ARTIFACT]  # outside if, needs to be run regardless
    return cell_dict


def make_type_dict(cell_list):
    types = sorted(set([cell.type for cell in cell_list]))
    return {
        cell_type: [cell for cell in cell_list if cell.type == cell_type]
        for cell_type in types
    }


def pretty_xml(elem, indentation_str="  "):
    ugly_xml = ElementTree.tostring(elem, "utf-8")
    md_parsed = minidom.parseString(ugly_xml)
    return md_parsed.toprettyxml(indent=indentation_str, encoding="UTF-8")


# TRANSCODE
def transform_xml_file(xml_file_path, output_file_path, transform_params):
    cells = get_cells(xml_file_path)  # TODO: check if cells_only
    for cell in cells:
        cell.transform(*transform_params)
    cells_to_xml(cells, output_file_path)


def find_relevant_tiffs(tiffs, cell_def):
    cells = [UntypedCell(tiff) for tiff in tiffs]
    if os.path.isdir(cell_def):
        relevant_cells = set(
            [UntypedCell(pos_from_file_name(f)) for f in os.listdir(cell_def)]
        )
    else:
        relevant_cells = set(
            [UntypedCell.from_cell(cell) for cell in get_cells(cell_def)]
        )
    return [
        tiffs[pos]
        for (pos, cell) in enumerate(cells)
        if cell in relevant_cells
    ]
