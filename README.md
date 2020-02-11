# Transformer Flir2Tif

IR (infrared) Image Bin to GeoTIFF Converter.

Plot level summaries are named ['surface_temperature'](http://mmisw.org/ont/cf/parameter/surface_temperature) in the trait database.
This name from the Climate Forecast (CF) conventions, and is used instead of 'canopy_temperature' for two reasons.
First, because we do not (currently) filter soil in this pipeline.
Second, because the CF definition of surface_temperature distinguishes the surface from the medium: "The surface temperature is the temperature at the interface, not the bulk temperature of the medium above or below."   http://cfconventions.org/Data/cf-standard-names/48/build/cf-standard-name-table.html

### Sample Docker Command line
Below is a sample command line that shows how the flir2tif Docker image could be run.
An explanation of the command line options used follows.
Be sure to read up on the [docker run](https://docs.docker.com/engine/reference/run/) command line for more information.

The data files used in this example are available on [Google Drive](https://drive.google.com/file/d/1AZT2S3yajitMCanIaQvnFra1XIbwYc5d/view?usp=sharing).

```docker run --rm --mount "src=/home/test,target=/mnt,type=bind" agpipeline/flir2tif:2.1 --metadata "/mnt/e475911c-3f79-4ebb-807f-f623d5ae7783_metadata_cleaned.json" --working_space "/mnt" "/mnt/e475911c-3f79-4ebb-807f-f623d5ae7783_ir.bin"```

This example command line assumes the source files are located in the `/home/test` folder of the local machine.
The name of the image to run is `agpipeline/flir2tif:2.1`.

We are using the same folder for the source files and the output files.
By using multiple `--mount` options, the source and output files can be located in separate folders.

**Docker commands** \
Everything between 'docker' and the name of the image are docker commands.

- `run` indicates we want to run an image
- `--rm` automatically delete the image instance after it's run
- `--mount "src=/home/test,target=/mnt,type=bind"` mounts the `/home/test` folder to the `/mnt` folder of the running image

We mount the `/home/test` folder to the running image to make files available to the software in the image.

**Image's commands** \
The command line parameters after the image name are passed to the software inside the image.
Note that the paths provided are relative to the running image (see the --mount option specified above).

- `--working_space "/mnt"` specifies the folder to use as a workspace
- `--metadata "/mnt/e475911c-3f79-4ebb-807f-f623d5ae7783_metadata_cleaned.json"` is the name of the cleaned metadata
- `"/mnt/e475911c-3f79-4ebb-807f-f623d5ae7783_ir.bin"` is the name of the raw image to convert
