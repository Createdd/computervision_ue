
## Main problem /tasks




- How to incorporate temporal images
Simply warping them again sufficient?

- What type of algorithm to use for anomaly detection
Pytorch from scratch or using pre-trained version from official JKU implementation (using other images)
AOS Enhanced Color Anomaly Detection (from research paper)

- Object detection what type of anomaly is it?







## Literature Overview

- nice overview: https://paperswithcode.com/task/unsupervised-anomaly-detection




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


---

- Normality-Calibrated Autoencoder for Unsupervised Anomaly Detection on Data Contamination -https://arxiv.org/abs/2110.14825v1
  - In this paper, we propose Normality-Calibrated Autoencoder (NCAE), which can boost anomaly detection performance on the contaminated datasets without any prior information or explicit abnormal samples in the training phase. The NCAE adversarially generates high confident normal samples from a latent space having low entropy and leverages them to predict abnormal samples in a training dataset

- A Unified Survey on Anomaly, Novelty, Open-Set, and Out-of-Distribution Detection: Solutions and Future Challenges - https://arxiv.org/pdf/2110.14051.pdf
  - To date, several research domains tackle the problem of detecting unfamiliar samples, including anomaly detection, novelty detection, one-class learning, open set recognition, and out-of-distribution detection. Despite having similar and shared concepts, out-of-distribution, open-set, and anomaly detection have been investigated independently. This survey aims to provide a cross-domain and comprehensive review of numerous eminent works in respective areas while identifying their commonalities.
  - Outlier or novelty detection? Our challenge illustrates a great example for anomaly detection, as our data has mixed outliers within. (training data = polluted with outliers)
  - To find a sample that deviates from the trend, adopting an appropriate distance metric is necessary. For instance, deviation could be computed in a raw pixel-level input or in a semantic space that is learned through a deep neural network. Some samples might have a low deviation from others in the raw pixel space but exhibit large deviations in representation space. Therefore, choosing the right distance measure for a hypothetical space is another challenge. Finally, the last challenge is choosing the threshold to determine whether the deviation from normal samples is significant.

- UAV-YOLO: Small Object Detection on Unmanned Aerial Vehicle Perspective - https://www.researchgate.net/publication/340708049_UAV-YOLO_Small_Object_Detection_on_Unmanned_Aerial_Vehicle_Perspective
  - Object detection, as a fundamental task in computer vision, has been developed enormously, but is still challenging work, especially for Unmanned Aerial Vehicle (UAV) perspective due to small scale of the target. 
  - In this study, the authors develop a special detection method for small objects in UAV perspective. Based on YOLOv3, the Resblock in darknet is first optimized by concatenating two ResNet units that have the same width and height. Then, the entire darknet structure is improved by increasing convolution operation at an early layer to enrich spatial information. Both these two optimizations can enlarge the receptive filed. 
  - Furthermore, UAV-viewed dataset is collected to UAV perspective or small object detection. An optimized training method is also proposed based on collected UAV-viewed dataset. The experimental results on public dataset and our collected UAV-viewed dataset show distinct performance improvement on small object detection with keeping the same level performance on normal dataset, which means our proposed method adapts to different kinds of conditions.

- A Comprehensive Approach for UAV Small Object Detection with Simulation-based Transfer Learning and Adaptive Fusion - https://arxiv.org/abs/2109.01800
  - Precisely detection of Unmanned Aerial Vehicles(UAVs) plays a critical role in UAV defense systems. Deep learning is widely adopted for UAV object detection whereas researches on this topic are limited by the amount of dataset and small scale of UAV. 
  - To tackle these problems, a novel comprehensive approach that combines transfer learning based on simulation data and adaptive fusion is proposed. Firstly, the open-source plugin AirSim proposed by Microsoft is used to generate mass realistic simulation data. Secondly, transfer learning is applied to obtain a pre-trained YOLOv5 model on the simulated dataset and fine-tuned model on the real-world dataset. Finally, an adaptive fusion mechanism is proposed to further improve small object detection performance. Experiment results demonstrate the effectiveness of simulation-based transfer learning which leads to a 2.7% performance increase on UAV object detection. Furthermore, with transfer learning and adaptive fusion mechanism, 7.1% improvement is achieved compared to the original YOLO v5 model.

- Rakesh John Amala Arokia Nathan, Indrajit Kurmi, David C. Schedl and Oliver Bimber, Through-Foliage Tracking with Airborne Optical Sectioning, Remote Sensing of Environment (under Review), (2021), https://arxiv.org/abs/2111.06959
  - Second, we show that color anomaly detection (e.g., Reed and Yu (1990); Ehret, Davy, Morel and Delbracio (2019)) benefits significantly from AOS integral images when compared to conventional single images (on average 97% vs. 42% in precision). Color anomaly detection is often used for automatized aerial image analysis in search and rescue applications (e.g., Morse, Thornton and Goodrich (2012); Agcayazi, Cawi, Jurgenson, Ghassemi and Cook (2016); Weldon and Hupy (2020))
- Indrajit Kurmi, David C. Schedl, and Oliver Bimber, Combined People Classification with Airborne Optical Sectioning, Nature Scientific Reports (under review), (2021), https://arxiv.org/abs/2106.10077
David C. Schedl, Indrajit Kurmi, and Oliver Bimber, Autonomous Drones for Search and Rescue in Forests, Science Robotics 6(55), eabg1188, https://doi.org/10.1126/scirobotics.abg1188, (2021)
- David C. Schedl, Indrajit Kurmi, and Oliver Bimber, Search and rescue with airborne optical sectioning, Nature Machine Intelligence 2, 783-790, https://doi.org/10.1038/s42256-020-00261-3 (2020)
- Indrajit Kurmi, David C. Schedl, and Oliver Bimber, Pose Error Reduction for Focus Enhancement in Thermal Synthetic Aperture Visualization, IEEE Geoscience and Remote Sensing Letters, DOI: https://doi.org/10.1109/LGRS.2021.3051718 (2021).
- Indrajit Kurmi, David C. Schedl, and Oliver Bimber, Fast automatic visibility optimization for thermal synthetic aperture visualization, IEEE Geosci. Remote Sens. Lett. https://doi.org/10.1109/LGRS.2020.2987471 (2020).
- David C. Schedl, Indrajit Kurmi, and Oliver Bimber, Airborne Optical Sectioning for Nesting Observation. Sci Rep 10, 7254, https://doi.org/10.1038/s41598-020-63317-9 (2020).
Indrajit Kurmi, David C. Schedl, and Oliver Bimber, Thermal airborne optical sectioning. Remote Sensing. 11, 1668, https://doi.org/10.3390/rs11141668, (2019).


