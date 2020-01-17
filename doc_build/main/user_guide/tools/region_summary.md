# Summarise cell counts from multiple brains

To summarise and organise the results from `cellfinder_cells_into_region` 
(which is a list of every brain region and the number of cells) you can use 
`cellfinder_region_summary`. 

This takes a directory of csv's output from 
`cellfinder_cells_into_region` and generates summary versions in which:
 * Brain regions are organised in the same way for each csv file
 * (Optionally, only the brain regions specified by `--regions-list` and/or 
 `--regions`)


``` bash
cellfinder_region_summary /path/to/directory/with/csvs
```

### Arguments
#### Mandatory
*  Directory containing csv files to be summarised

#### The following options can also be used:
* `--regions-list` A text file listing a subset of brain regions to be included
 in the summary csv file (as per the allen brain atlas csv)
 * `--sum-regions` Rather than finding all child regions, sum them.
* `--regions` A list of additional regions to include, e.g:
```bash
--regions "Anterior cingulate area" "Retrosplenial area"
``` 
* `--structures-file` The csv file containing the structures definition
 (if not using the default Allen brain atlas) **[NOT YET IMPLEMENTED]**

N.B. The regions are hierarchical, so if you specify 
`Retrosplenial area, dorsal part`, 
you will get all subregions:
* Retrosplenial area, dorsal part
* Retrosplenial area, dorsal part, layer 1
* Retrosplenial area, dorsal part, layer 2/3
* Retrosplenial area, dorsal part, layer 4
* Retrosplenial area, dorsal part, layer 5
* Retrosplenial area, dorsal part, layer 6a
* Retrosplenial area, dorsal part, layer 6b
