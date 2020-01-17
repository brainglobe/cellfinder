from anytree import AnyNode


class BrainStructure(AnyNode):
    def __init__(
        self,
        _id,
        atlas_id,
        name,
        acronym,
        st_level,
        ontology_id,
        hemisphere_id,
        weight,
        _parent_structure_id,
        _depth,
        graph_id,
        graph_order,
        _structure_id_path,
        color_hex_triplet,
        neuro_name_structure_id,
        neuro_name_structure_id_path,
        failed,
        sphinx_id,
        structure_name_facet,
        failed_facet,
        safe_name,
        parent=None,
    ):
        super(BrainStructure, self).__init__()
        self.id = _id  # The numerical id
        self.atlas_id = atlas_id  # ? missing for some
        self.name = name  # The human readable name
        self.acronym = acronym  # The standard acronym of the structure
        self.st_level = st_level  # None
        self.ontology_id = ontology_id  # constant
        self.hemisphere_id = hemisphere_id  # constant
        self.weight = weight  # constant
        # The numerical id (self.id) of the parent structure (property)
        self._parent_structure_id = _parent_structure_id
        # The depth in of this structure in the structure tree (property)
        # self._depth = depth
        self.graph_id = graph_id  # constant
        self.graph_order = graph_order  # ?
        # The whole path in the tree (property)
        self._structure_id_path = _structure_id_path
        # The color to represent in the atlas (hex triplet)
        self.color_hex_triplet = color_hex_triplet
        self.neuro_name_structure_id = neuro_name_structure_id  # None
        # None
        self.neuro_name_structure_id_path = neuro_name_structure_id_path
        self.failed = failed  # constant 'f'
        self.sphinx_id = sphinx_id  # ?
        self.structure_name_facet = structure_name_facet  # ?
        self.failed_facet = failed_facet  # ?
        self.safe_name = safe_name  # ?

    @property
    def parent_structure_id(self):
        return self.parent.id

    @property
    def structure_id_path(self):
        return self.path
