# XML file cropping

To "crop" an xml file, i.e. only include cell coordinates corresponding to 
given anatomical areas.


```bash
    cellfinder_xml_crop --xml-dir xml_directory --structures-file chosen_structures.csv --registered-atlas registered_atlas.nii --hemispheres hemispheres.nii

```

### Arguments
Run `cellfinder_xml_crop -h` to see all options.

#### Mandatory
* `--xml-dir` Directory containing xml files to be cropped
* `--structures-file` Curated csv structure list (as per the allen
 brain atlas csv
 * `--registered-atlas` The path to the atlas registered to the sample brain
* `--hemispheres` The atlas with just the hemispheres encoded


#### The following options may also need to be used:
* `--hemisphere-query` Which hemisphere to keep (1 or 2). Default: 0 (all)

**Either**
* `-x` or `--x-pixel-mm` Pixel spacing of the data in the first dimension, 
specified in mm.
* `-y` or `--y-pixel-mm` Pixel spacing of the data in the second dimension, 
specified in mm.
* `-z` or `--z-pixel-mm` Pixel spacing of the data in the third dimension, 
specified in mm.

**Or**
* `--metadata` Metadata file containing pixel sizes (supported formats 
include [BakingTray](https://github.com/SainsburyWellcomeCentre/BakingTray) 
recipe files [mesoSPIM](https://github.com/mesoSPIM/mesoSPIM-control) 
 metadata files or 
 [cellfinder custom metadata files](https://github.com/SainsburyWellcomeCentre/cellfinder/tree/master/doc_build/examples/cellfinder_metadata.ini).
  If both pixel sizes and metadata are provided, the command line arguments 
  will take priority.