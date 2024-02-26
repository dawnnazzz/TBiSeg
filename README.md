## Dataset used in TBiSeg
We constructed an integrated inland waterways semantic segmentation dataset, which consists of the USVInland-Water-Segmentation dataset and the Tampere-WaterSeg Dataset:

- [the USVInland-Water-Segmentation dataset](https://orca-tech.cn/datasets/USVInland/Waterline)
  
- [the Tampere-WaterSeg dataset](https://etsin.fairdata.fi/dataset/e0c6ef65-6e1e-4739-abe3-0455697df5ab)
  
The integrated dataset consists of 1400 images, which encompass scenes that USVs may encounter when navigating autonomously on inland waterways, including reflections, fog, rain, irregular waterline conditions, and different-sized obstacles. All annotations have been unified, where 0 represents background and 1 represents water, and can be trained in a semantic segmentation model directly.
