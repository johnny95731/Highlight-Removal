# Highlight-Removal
Summary of Publicly Available Highlight Removal Method

**Raise Issue or PR to add more**

## Dataset

[MIT Intrinsic Images](https://www.cs.toronto.edu/~rgrosse/intrinsic/)

[SSHR](https://github.com/fu123456/TSHRNet)

[SHIQ](https://github.com/fu123456/SHIQ)

[PSD](https://github.com/jianweiguo/SpecularityNet-PSD)

[kvasir](https://datasets.simula.no/kvasir/)

[WHU-Specular](https://github.com/fu123456/SHDNet)

[WHU-TRIIW](https://github.com/fu123456/SHDNet)

[EndoSRR](https://github.com/Tobyzai/EndoSRR)

[Hyper Kvasir](https://github.com/RemaDaher/CycleSTTN/blob/main/dataset_prep/README.md)

## Method

### Traditional

| Authors | Year | Publication | Code |
| ------- | ---- | ----------- | ---- |
| Lin et al. [1] | 2003 | ICCV | [MATLAB](https://github.com/Kanvases/Highlight-Removal-by-Illumination-Constrained-Inpainting) |
| Tan et al. [2] | 2005 | TPAMI | [MATLAB](https://github.com/vitorsr/SIHR) |
| Yoon et al. [3] | 2006 | ICIP | [MATLAB](https://github.com/vitorsr/SIHR) |
| Shen et al. [4] | 2008 | Pattern Recognition | [MATLAB](https://github.com/vitorsr/SIHR) |
| Shen et al. [5] | 2009 | Applied Optics | [MATLAB](https://github.com/vitorsr/SIHR) |
| Yang et al. [6] | 2010 | ECCV | [MATLAB](https://github.com/vitorsr/SIHR) |
| Arnold et al. [7] | 2010 | Journal on Image and Video Processing | [MATLAB](https://github.com/jiemojiemo/some_specular_detection_and_inpainting_methods_for_endoscope_image) |
| Saint-Pierre et al. [8] | 2011 | Machine Vision and Applications | [MATLAB](https://github.com/jiemojiemo/some_specular_detection_and_inpainting_methods_for_endoscope_image) |
| El Meslouhi et al. [9] | 2011 | Open Computer Science | [MATLAB](https://github.com/jiemojiemo/some_specular_detection_and_inpainting_methods_for_endoscope_image) |
| Shen et al. [10] | 2013 | Applied Optics | [MATLAB](https://github.com/vitorsr/SIHR) |
| Huo et al. [11] | 2015 | ICMEW | [Python](https://github.com/EthanWooo/Huo15) |
| Akashi et al. [12] | 2016 | CVIU | [MATLAB](https://github.com/vitorsr/SIHR) |
| Nurutdinova et al. [13] | 2017 | VISAPP | [C++](https://github.com/AlexandraPapadaki/Specularity-Shadow-and-Occlusion-Removal-for-Planar-Objects-in-Stereo-Case) |
| Souza et al. [14] | 2018 | SIBGRAPI | [C++](https://github.com/MarcioCerqueira/RealTimeSpecularHighlightRemoval) |
| Banik et al. [15] | 2018 | ICCE | [Python](https://github.com/EthanWooo/Banik181) |
| Zhang et al. [16] | 2019 | CGF |  |
| Yamamoto et al. [17] | 2019 | ITE Transactions on Media Technology and Applications | [MATLAB](https://github.com/vitorsr/SIHR)                    |
| Li et al. [18] | 2020 | TMI | [C++](https://github.com/CXH-Research/Highlight-Removal/tree/main/Highlight-Removal) |
| Saha et al. [19] | 2020 | IET Image Processing | [MATLAB](https://github.com/7ZFG1/Combining-Highlight-Re-moval-and-Lowlight-Image-Enhancement-Technique-for-HDR-like-Image-Generation-) |
| Wen et al. [20] | 2021 | TIP | [MATLAB](https://github.com/wsj890411/Polar_HR)              |
| Pan et al. [21] | 2023 | CMPB | [C++](https://github.com/CXH-Research/Highlight-Removal/tree/main/Highlight-Removal) |
| Guo et al. [22] | 2017 | TIP | [Python](https://github.com/aeinrw/LIME/tree/master) |

### Learning

| Name | Year | Publication | Code |
| ---- | ---- | ----------- | ---- |
| SLRR | 2018 | ECCV | [Pytorch](https://github.com/dingguanglei/SLRR-SparseAndLowRankReflectionModel) |
| Spec-CGAN | 2020 | Image and Vision Computing | [Tensorflow](https://github.com/msiraj83/SPEC-CGAN) |
| TASHR | 2021 | PRCV | [Tensorflow](https://github.com/weizequan/TASHR) |
| SpecularityNet | 2021 | TMM | [Pytorch](https://github.com/jianweiguo/SpecularityNet-PSD) |
| JSHDR | 2021 | CVPR | [Pytorch](https://github.com/fu123456/SHIQ) |
| Liang et al. | 2021 | Optics Express | [Pytorch](https://github.com/Deepyanyuan/FaceIntrinsicDecomposition) |
| Unet-Transformer | 2022 | CVM | [Pytorch](https://github.com/hzfengfengxia/specularityRemoval) |
| MG-CycleGAN | 2022 | PRL | [Pytorch](https://github.com/hootoon/MG-Cycle-GAN) |
| TSRNet | 2023 | ICCV | [Pytorch](https://github.com/fu123456/TSHRNet) |
| SHMGAN | 2023 | Neurocomputing | [Tensorflow](https://github.com/Atif-Anwer/SHMGAN) |
| CycleSTTN | 2023 | MICCAI | [Pytorch](https://github.com/RemaDaher/CycleSTTN) |
| Endo-STTN | 2023 | MIA | [Pytorch](https://github.com/endomapper/Endo-STTN) |
| MEF-SHDR | 2023 | PRCV | [Pytorch](https://github.com/ErisGe/MEF-SHDR) |
| | 2024 | ICASSP | [Pytorch](https://github.com/LittleFocus2201/ICASSP2024) |
| EndoSRR | 2024 | International Journal of Computer Assisted Radiology and Surgery | [Pytorch](https://github.com/Tobyzai/EndoSRR) |
| Film Removal | 2024 | CVPR | [Pytorch](https://github.com/jqtangust/filmremoval) |
| HighlightRemover | 2024 | MM | [Pytorch](https://github.com/zz0223/HRNet) |
| DHAN-SHR | 2024 | MM | [Pytorch](https://github.com/CXH-Research/DHAN-SHR) |
| ORHLR-Net | 2024 | TVC | [Pytorch](https://github.com/hzq2333/ORHLR-Net) |
| FHR-Net | 2024 | TCSVT | [Pytorch](https://github.com/hongsheng-Z/FHR-Net) |

## Metric

### Full-Reference

[torchmetrics](https://github.com/Lightning-AI/torchmetrics) for cuda calculation


## References

[1] PingTan, Lin, Long Quan and Heung-Yeung Shum, "Highlight removal by illumination-constrained inpainting," Proceedings Ninth IEEE International Conference on Computer Vision, Nice, France, 2003, pp. 164-169 vol.1, doi: [10.1109/ICCV.2003.1238333](http://doi.org/10.1109/ICCV.2003.1238333)

[2] Artusi, A., Banterle, F. and Chetverikov, D., "A Survey of Specularity Removal Methods," Computer Graphics Forum, 2011, 30: 2208-2230., doi: [10.1111/j.1467-8659.2011.01971.x](https://doi.org/10.1111/j.1467-8659.2011.01971.x)

[3] K. -j. Yoon, Y. Choi and I. S. Kweon, "Fast Separation of Reflection Components using a Specularity-Invariant Image Representation," 2006 International Conference on Image Processing, Atlanta, GA, USA, 2006, pp. 973-976, doi: [10.1109/ICIP.2006.312650](http://doi.org/10.1109/ICIP.2006.312650)

[4] Hui-Liang Shen, Hong-Gang Zhang, Si-Jie Shao, John H. Xin, "Chromaticity-based separation of reflection components in a single image," Pattern Recognition, Volume 41, Issue 8, 2008, pp. 2461-2469, doi: [10.1016/j.patcog.2008.01.026](https://www.sciencedirect.com/science/article/pii/S0031320308000654)

[5] Hui-Liang Shen and Qing-Yuan Cai, "Simple and efficient method for specularity removal in an image," Appl. Opt., Vol. 48, Issue 14, 2009, pp. 2711-2719, doi: [10.1364/AO.48.002711](https://doi.org/10.1364/AO.48.002711)

[6] Yang, Q., Wang, S., Ahuja, N. "Real-Time Specular Highlight Removal Using Bilateral Filtering," Computer Vision – ECCV 2010, vol 6314, pp. 87-100, doi: [10.1007/978-3-642-15561-1_7](https://doi.org/10.1007/978-3-642-15561-1_7)

[7] Arnold, M., Ghosh, A., Ameling, S. et al. Automatic Segmentation and Inpainting of Specular Highlights for Endoscopic Imaging. J Image Video Proc 2010, 814319 (2010). https://doi.org/10.1155/2010/814319

[8] Charles-Auguste Saint-Pierre, Jonathan Boisvert, Guy Grimard, and Farida Cheriet. 2011. Detection and correction of specular reflections for automatic surgical tool segmentation in thoracoscopic images. Mach. Vision Appl. 22, 1 (January 2011), 171–180. https://dl.acm.org/doi/abs/10.5555/2150916.2150919

[9] El Meslouhi, O., Kardouchi, M., Allali, H. et al. Automatic detection and inpainting of specular reflections for colposcopic images. centr.eur.j.comp.sci. 1, 341–354 (2011). https://doi.org/10.2478/s13537-011-0020-2

[10] Hui-Liang Shen and Zhi-Huan Zheng, "Real-time highlight removal using intensity ratio," Appl. Opt. 52, 4483-4493 (2013) https://opg.optica.org/ao/abstract.cfm?uri=ao-52-19-4483

[11] Yongqing Huo, Fan Yang and Chao Li, "HDR image generation from LDR image with highlight removal," 2015 IEEE International Conference on Multimedia & Expo Workshops (ICMEW), Turin, 2015, pp. 1-5, doi: [10.1109/ICMEW.2015.7169849](https://doi.org/10.1109/ICMEW.2015.7169849)

[12] Yasushi Akashi, Takayuki Okatani, "Separation of reflection components by sparse non-negative matrix factorization," Computer Vision and Image Understanding, Vol 146, 2016, pp. 77-85, dor: [10.1016/j.cviu.2015.09.001](https://doi.org/10.1016/j.cviu.2015.09.001)

[13] Nurutdinova, I., Hänsch, R., Mühler, V., Bourou, S., I. Papadaki, A. and Hellwich, O. (2017). Specularity, Shadow, and Occlusion Removal for Planar Objects in Stereo Case. In Proceedings of the 12th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP 2017) - Volume 4: VISAPP; ISBN 978-989-758-225-7; ISSN 2184-4321, SciTePress, pages 98-106. DOI: [10.5220/0006166100980106](https://doi.org/10.5220/0006166100980106)

[14] A. C. S. Souza, M. C. F. Macedo, V. P. Nascimento and B. S. Oliveira, "Real-Time High-Quality Specular Highlight Removal Using Efficient Pixel Clustering," 2018 31st SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI), Parana, Brazil, 2018, pp. 56-63, doi: [10.1109/SIBGRAPI.2018.00014](https://doi.org/10.1109/SIBGRAPI.2018.00014)

[15] P. P. Banik, R. Saha and K. -D. Kim, "HDR image from single LDR image after removing highlight," 2018 IEEE International Conference on Consumer Electronics (ICCE), Las Vegas, NV, USA, 2018, pp. 1-4, doi: [10.1109/ICCE.2018.8326106](https://doi.org/10.1109/ICCE.2018.8326106)

[16] Fu, G., Zhang, Q., Song, C., Lin, Q. and Xiao, C. (2019), Specular Highlight Removal for Real-world Images. Computer Graphics Forum, 38: 253-263. https://doi.org/10.1111/cgf.13834

[17] T. Yamamoto and A. Nakazawa, “General Improvement Method of Specular Component Separation Using High-Emphasis Filter and Similarity Function,” ITE Transactions on Media Technology and Applications, 2019, vol. 7, no. 2, pp. 92–102
[10.3169/mta.7.92](https://doi.org/10.3169/mta.7.92)

[18] R. Li, J. Pan, Y. Si, B. Yan, Y. Hu and H. Qin, "Specular Reflections Removal for Endoscopic Image Sequences With Adaptive-RPCA Decomposition," in IEEE Transactions on Medical Imaging, vol. 39, no. 2, pp. 328-340, Feb. 2020, doi: [10.1109/TMI.2019.2926501](https://dot.org/10.1109/TMI.2019.2926501)

[19] Saha, R., Pratim Banik, P., Sen Gupta, S. and Kim, K.-D. (2020), Combining highlight removal and low-light image enhancement technique for HDR-like image generation. IET Image Processing, 14: 1851-1861. https://doi.org/10.1049/iet-ipr.2019.1099

[20] S. Wen, Y. Zheng and F. Lu, "Polarization Guided Specular Reflection Separation," in IEEE Transactions on Image Processing, vol. 30, pp. 7280-7291, 2021, doi: [10.1109/TIP.2021.3104188](https://doi.org/10.1109/TIP.2021.3104188)

[21] Junjun Pan, Ranyang Li, Hongjun Liu, Yong Hu, Wenhao Zheng, Bin Yan, Yunsheng Yang, Yi Xiao, "Highlight removal for endoscopic images based on accelerated adaptive non-convex RPCA decomposition," Computer Methods and Programs in Biomedicine, Volume 228, 2023, 107240, https://doi.org/10.1016/j.cmpb.2022.107240.

[22] Guo X, Li Y, Ling H. LIME: Low-Light Image Enhancement via Illumination Map Estimation. IEEE Transactions on Image Processing 2017, 26 (2), 982–993. https://doi.org/10.1109/TIP.2016.2639450.