# Sakura

A GPU renderer written with CUDA, initially for the Computer Graphics lecture at ETH Zurich.


## Usage

First clone the repository and initialize the submodules using `git submodule update --init --recursive`.

If all the prerequisites are installed, the executable can be generated using the following commands:
```
mkdir build
cd build
cmake ..
make 
```

or `make -j N` to make use of N threads

[//]: # (## Example Images)

[//]: # (![CornellBox]&#40;renders/CBOX4K512spp.png "Cornell Box render"&#41;)


[//]: # (## Special Features)

[//]: # ()
[//]: # (#### Adaptive Sampling)

[//]: # (|                Test Scene                |               Sample Distribution               |)

[//]: # (|:----------------------------------------:|:-----------------------------------------------:|)

[//]: # (| ![]&#40;renders/images/adaptiveSampling.png&#41; | ![]&#40;renders/images/adaptiveSamplingSamples.png&#41; |)
