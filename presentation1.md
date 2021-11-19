
## Main problem /tasks




How to incorporate temporal images
Simply warping them again sufficient?

What type of algorithm to use for anomaly detection
Pytorch from scratch or using pre-trained version from official JKU implementation (using other images)

Object detection what type of anomaly is it?







## Literature Overview




- SaRNet: A Dataset for Deep Learning Assisted Search and Rescue with Satellite Imagery - https://arxiv.org/abs/2107.12469
Exploring similar dataset and its usage

- Small Target Detection for Search and Rescue Operations using Distributed Deep Learning and Synthetic Data Generation - https://arxiv.org/pdf/1904.11619.pdf
“We combined image segmentation, enhancement, and convolution neural networks to reduce detection time to detect small targets. We compared the performance between the auto-detection system and the human eye. Our system detected the target within 8 seconds, but the human eye detected the target within 25 seconds. Our systems also used synthetic data generation and data augmentation techniques to improve target detection accuracy”
- Automatic Person Detection in Search and Rescue Operations Using Deep CNN Detectors - https://ieeexplore.ieee.org/document/9369386/algorithms
 In this paper, the reliability of existing state-of-the-art detectors such as Faster R-CNN, YOLOv4, RetinaNet, and Cascade R-CNN on a VisDrone benchmark and custom-made dataset SARD build to simulate rescue scenes was investigated. After training the models on selected datasets, detection results were compared. Because of the high speed and accuracy and the small number of false detections, the YOLOv4 detector was chosen for further examination. […] YOLOv4 has achieved the best detection performances 

- Deep Reinforcement Learning for Autonomous Search and Rescue - https://ieeexplore.ieee.org/document/8556642
The prototype successfully demonstrated the feasibility of using an artificial intelligence to direct unmanned aerial vehicles to search. […] However, given the real-time, real-physics nature of a single simulated run, training time simply takes too long, inhibiting the success rate of the intelligent system




- Search and Rescue with Airborne Optical Sectioning - https://arxiv.org/abs/2009.08835
We show that automated person detection under occlusion conditions can be significantly improved by combining multiperspective images before classification. Here, we employed image integration by Airborne Optical Sectioning (AOS)—a synthetic aperture imaging technique that uses camera drones to capture unstructured thermal light fields—to achieve this with a precision/recall of 96/93%. Finding lost or injured people in dense forests is not generally feasible with thermal recordings, but becomes practical with use of AOS integral images.
  - https://github.com/JKU-ICG/AOS

- Through-Foliage Tracking with Airborne Optical Sectioning - https://arxiv.org/abs/2111.06959
While detecting and tracking moving targets through foliage is difficult (and for many cases even impossible) in regular aerial images or videos, it becomes practically feasible with image integration – which is the core principle of Airborne Optical Sectioning. We have already shown that classification significantly benefits from image integration (Schedl et al. (2020b)). In this work we demonstrate that the same holds true for color anomaly detection.





Rakesh John Amala Arokia Nathan, Indrajit Kurmi, David C. Schedl and Oliver Bimber, Through-Foliage Tracking with Airborne Optical Sectioning, Remote Sensing of Environment (under Review), (2021), https://arxiv.org/abs/2111.06959
 Indrajit Kurmi, David C. Schedl, and Oliver Bimber, Combined People Classification with Airborne Optical Sectioning, Nature Scientific Reports (under review), (2021), https://arxiv.org/abs/2106.10077
David C. Schedl, Indrajit Kurmi, and Oliver Bimber, Autonomous Drones for Search and Rescue in Forests, Science Robotics 6(55), eabg1188, https://doi.org/10.1126/scirobotics.abg1188, (2021)
David C. Schedl, Indrajit Kurmi, and Oliver Bimber, Search and rescue with airborne optical sectioning, Nature Machine Intelligence 2, 783-790, https://doi.org/10.1038/s42256-020-00261-3 (2020)
Indrajit Kurmi, David C. Schedl, and Oliver Bimber, Pose Error Reduction for Focus Enhancement in Thermal Synthetic Aperture Visualization, IEEE Geoscience and Remote Sensing Letters, DOI: https://doi.org/10.1109/LGRS.2021.3051718 (2021).
Indrajit Kurmi, David C. Schedl, and Oliver Bimber, Fast automatic visibility optimization for thermal synthetic aperture visualization, IEEE Geosci. Remote Sens. Lett. https://doi.org/10.1109/LGRS.2020.2987471 (2020).
David C. Schedl, Indrajit Kurmi, and Oliver Bimber, Airborne Optical Sectioning for Nesting Observation. Sci Rep 10, 7254, https://doi.org/10.1038/s41598-020-63317-9 (2020).
Indrajit Kurmi, David C. Schedl, and Oliver Bimber, Thermal airborne optical sectioning. Remote Sensing. 11, 1668, https://doi.org/10.3390/rs11141668, (2019).


