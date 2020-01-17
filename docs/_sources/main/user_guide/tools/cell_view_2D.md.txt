# Two dimensional cell viewer

To visualise cells (or cell candidates) from an XML file overlaid onto a
full-resolution image. Planes are loaded only when needed, so uses
very little ram.


``` bash
    cellfinder_view_cells /path/to/xmlfile.xml /path/to/images.txt
```

### Arguments
#### Mandatory
* `-i` or `--img-paths` Directory of images
* `-x` or `--cells-xml` Path to the .xml cell file
 
A viewer will then open, showing one 2D slice at a time, with the cells
overlaid as circles. A slider at the bottom can be used to navigate the stack,
and the image can be zoomed in using the mouse scroll wheel.

