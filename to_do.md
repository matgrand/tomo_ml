# Stuff to do

- [v] Make all the inputs scale independent, so that the network can be trained on inputs between 0 and 1, much better + a scale factor. In theory both the input and output should be scalabale by the same constant
- [v] Make synthetic SXR evaluation (line integration) easier, by loading the masks from a file, not recalculating them every time
- [ ] Tune RFX parameters to get a closer to realistic SXR values
- [ ] Normalize each of the inputs substracting the mean and dividing by the standard deviation, make an analysis of each input of each SXR (mean, std, min, max) and build a syhtnetic dataset sampling from the same distribution