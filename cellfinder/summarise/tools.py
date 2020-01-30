class CellCountMissingCellsException(Exception):
    pass


class UnknownAtlasValue(Exception):
    pass


def atlas_value_to_structure_id(atlas_value, structures_reference_df):
    line = structures_reference_df[
        structures_reference_df["id"] == atlas_value
    ]
    if len(line) == 0:
        raise UnknownAtlasValue(atlas_value)
    structure_id = line["structure_id_path"].values[0]
    return structure_id


def atlas_value_to_name(atlas_value, structures_reference_df):
    line = structures_reference_df[
        structures_reference_df["id"] == atlas_value
    ]
    if len(line) == 0:
        raise UnknownAtlasValue(atlas_value)
    name = line["name"]
    return str(name.values[0])
