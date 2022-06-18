# Master Thesis - Super-Resolution
Deep convolutional networks for super-resolution are an intensively researched method
in the field of computer vision. Applying these technologies to the field of remote sensing
can provide many benefits, mainly since a high temporal and a high spatial resolution are
mutually exclusive for remote sensing products. Super-resolution has the possibility of
bridging that gap by upsampling the high frequency imagery to a higher spatial resolution. Many challenges arise when trying to implement super-resolution techniques to the
specific characteristics of remote sensing data, as well as ensuring that the needs of the
remote sensing communities towards the resulting data are met.
This thesis compares how well several different models with different architectures can
adapt to the task of performing super-resolution for the region of Brittany in France, using
data from the SPOT-6 satellite as high-resolution ground truth to super-resolute Sentinel-2
imagery. We also demonstrate that employing a histogram matching approach can successfully bridge the spectral gap between the sensors. The experimentation shows that the
standard Super-Resolution Convolutional Neural Network (SRCNN) and Residual Channel Attention Network (RCAN) succeed in adapting the color mapping, but fail to deliver
realistic-appearing super-resolution images. On the other hand, we show that the SuperResolution Generative Adversarial Network (SRGAN) can produce visually impressive
plausible super-resoluted images, but that the products lose the connection to the underlying real-world phenomena that the satellite imagery depicts.
The method of including time-series information by a simple band-stacking approach
into the generative adversarial model is shown not to be sufficient, while the fusion of
encoded information of multi-temporal low-resolution imagery via a recursive network
shows promising potential. In order to build on that potential, the main challenges to
overcome are identified as formulating useful loss functions as well as the publishing of
a common dataset to provide a comparison baseline, which is geared towards a real-life
application.
