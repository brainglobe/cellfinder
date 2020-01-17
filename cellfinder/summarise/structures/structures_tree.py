import csv
import pandas as pd

from cellfinder.summarise.structures.structure import BrainStructure


class StructureNotFoundError(Exception):
    def __init__(self, _id):
        self._id = _id

    def __str__(self):
        return "Missing structure with id {}".format(self._id)


def load_structures(structures_file_path):
    with open(structures_file_path, "r") as structures_file:
        structures_reader = csv.reader(
            structures_file, delimiter=",", quotechar='"'
        )
        structures = list(structures_reader)
        header = structures[0]
        structures = structures[1:]
        return header, structures


def load_structures_as_df(structures_file_path):
    return pd.read_csv(structures_file_path, sep=",", header=0, quotechar='"')


def get_struct_by_id(structures, _id):
    for struct in structures:
        if struct.id == _id:
            return struct
    raise StructureNotFoundError(_id)


def get_structures_tree(structures_file_path):
    header, structures_data = load_structures(structures_file_path)
    structures = []
    for struct_data in structures_data:
        struct = BrainStructure(*struct_data)
        if structures:
            struct.parent = get_struct_by_id(
                structures, struct._parent_structure_id
            )
        structures.append(struct)
    return structures


def print_tree(structures_tree, root_node_idx=0):
    from anytree import RenderTree, AsciiStyle

    root_node = structures_tree[root_node_idx]
    print(RenderTree(root_node, style=AsciiStyle()))
