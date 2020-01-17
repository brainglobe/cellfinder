# Organise cells into atlas regions

To understand results of the cell classification, the cell coordinates
from the `cell_classification.xml` file must be "assigned" to brain regions
from the registration step.


``` bash
cellfinder_count_summary --registered-atlas /path/to/registered_atlas.nii --hemispheres /path/to/hemispheres.nii --xml /path/to/classified cells.xml --structures-file /path/to/structures.csv --csv-out /path/to/csv_out.csv
```

### Arguments
#### Mandatory
* `--registered-atlas` The path to the atlas registered to the sample brain
* `--hemispheres` The atlas with just the hemispheres encoded
* `--xml` The xml file containing the cell locations
* `----output-dir` Directory to write the results 

The registered atlases and the xml will be read, and a csv showing how many 
cells are in each brain area (per hemisphere) is generated