# Average-Image-Analysis
This project aims to provide a new perspective to understand the trained Convolutional Neural Network(CNN) model and features in its filters. Guided Backporpagation is a typical  way of CNN filter visualizaiton, which could generate visualization images for every filter according a given image. The general information of datasets could be represented as its average image. Therefore, comparing the distance of average image and filter visualization image could describe the general information extraction power of a filter. In this way, we could quantify the confusing visualization results as several scalars, called as SimValues. After computing the SimValues of all filters in all layers, we could analyse them by their absolute values, variances and changing trends to get more insights about CNN model.