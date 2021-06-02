# Mouse cam

An open-source, UV-green camera designed to match the spectral sensitivities of the the mouse photoreceptors and to record footage in the natural environment of mice. This repository contains scripts, notebooks and data links related to our recent publication:

Qiu Y, Zhao Z, Klindt D, Kautzky M, Szatko KP, Schaeffel F, Rifai K, Franke K, Busse L, Euler T (2021) _Natural environment statistics in the upper and lower visual field are reflected in mouse retinal specializations._ **Current Biology**, accepted.

## UV-green footage from the mouse habitat

The movies listed in the table below are available from [here](https://zenodo.org/record/4812404#.YK6dNbczapo) and can be accessed via the notebook in `footage`. 

[<img src="https://github.com/eulerlab/mouse-scene-cam/blob/master/pictures/table_mousefootage.png" alt="Table" width="640"/>](https://github.com/eulerlab/mouse-scene-cam/blob/master/pictures/table_mousefootage.png)


## Repository structure

```
├───pictures                       - Assorted pictures
├───photoisomerizations            - Notebook to estimate photoisomerization rates from camera images
│   └───data                       - Spectral data for notebook
└───printed_parts                  - .stl, .scad (OpenSCAD scripts)
└───code                           - Code for analysis
└───data                           - Data for analysis
│   └───mouse_footage              - All mouse movies
└───footage                        - Notebook to read footage data and save as .avi file
```

