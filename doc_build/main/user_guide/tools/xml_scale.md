# XML file rescaling

To rescale the cell positions within an XML file. For compatibility with other 
software, or if your data has been scaled after cell detection.

```bash
    cellfinder_xml_scale /path/to.xml -x x_scale -y y_scale -z z_scale
```

### Arguments
Run `cellfinder_xml_scale -h` to see all options.

#### Optional arguments
**Scales** 

If not set, will default to 1 (i.e. no scaling). Can be any float
 (e.g. 0.5 or 2)
* `-x` or `--x-scale` Rescaling factor in the first dimension
* `-y` or `--y-scale` Rescaling factor in the second dimension
* `-z` or `--z-scale` Rescaling factor in the third dimension

**Others**
* `-o` or `--output` Output directory for the scaled xml file. Defaults to the 
same directory as the input file. 
