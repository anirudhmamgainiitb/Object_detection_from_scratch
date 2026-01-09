### Understanding dataset:

nc: 7
names: ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']

Train samples : 448
Validation samples : 127
Test samples : 63

Train distribution: Counter({'fish': 1961, 'jellyfish': 385, 'penguin': 330, 'shark': 259, 'puffin': 175, 'stingray': 136, 'starfish': 78})
Validation distribution: Counter({'fish': 459, 'jellyfish': 155, 'penguin': 104, 'puffin': 74, 'shark': 57, 'stingray': 33, 'starfish': 27})
Test distribution: Counter({'fish': 249, 'jellyfish': 154, 'penguin': 82, 'shark': 38, 'puffin': 35, 'stingray': 15, 'starfish': 11})

### Data pre-processing and loading(No augmentations are used)

Input Resolution and Grid Design

Images are resized to 416×416, which aligns with the backbone’s downsampling factor and produces a 13×13 final feature map. This feature map is used as the detection grid, where each cell predicts bounding box coordinates, objectness, and class probabilities. This alignment ensures stable grid-based detection and consistent target assignment during training from scratch.

Strided Convolutions vs. Pooling

Strided convolutions are used for downsampling instead of pooling to allow the network to learn optimal spatial feature aggregation. This improves feature representation and training stability, particularly when no pretrained weights are used.


### Architechtural choice - Single stage detection using a custom cnn backbone and detection head

### Assumption:
each grid cell only corresponds to one object

The backbone network is a custom CNN trained from scratch without any pretrained weights. It consists of stacked convolutional blocks with 3×3 kernels, Batch Normalization, and ReLU activations. Strided convolutions are used for spatial downsampling instead of max pooling, allowing the network to learn optimal feature aggregation. The channel depth is progressively increased from 32 to 512 to capture increasingly abstract semantic features, while keeping the overall depth moderate to prevent overfitting on the underwater dataset

### detection head

Detection Head

The detection head is implemented as a 1×1 convolution applied to the final feature map produced by the backbone. For each spatial location (grid cell), the detector predicts:

1.Bounding box coordinates (x, y, width, height)
2.Objectness score (whether an object is present)
3.Class scores for the 7 underwater object classes

### loss functions

The model is trained using a composite loss that combines bounding box regression, objectness prediction, and classification.

1.Bounding box localization is learned using Complete IoU (CIoU) loss, which considers overlap, center distance, and aspect ratio between predicted and ground-truth boxes. This improves localization accuracy, especially for elongated and overlapping underwater objects.

2.Objectness is trained using Binary Cross-Entropy loss, with reduced weight for background cells to handle severe class imbalance and prevent the model from predicting background everywhere.

3.Classification is trained using Cross-Entropy loss, since each object belongs to exactly one of the seven classes. Classification loss is computed only for grid cells containing objects

### Evaluation of model(no augmentation)
#### map(iou) taken is 0.25

| Metric              | Value       |
| ------------------- | ----------- |
| mAP (overall)       | 0.000002    |
| mAP (large objects) | 0.000002    |
| mAR@1               | 0.0022      |
| mAR@10              | 0.0028      |
| mAR@100             | 0.0028      |
| Evaluated Classes   | 7           |

### inference 

| Device          | FPS    | Latency (ms/image) |
| --------------- | ------ | ------------------ |
| CPU             | 25.14  | 39.77              |
| Apple MPS (GPU) | 125.39 | 7.98               |

Model size - 6.03 MB

### Experiments using three different set of configurations

| OBJ Threshold | SCORE Threshold | mAP@0.25 | FPS (MPS)  | Avg Detections / Image |
| ------------- | --------------- | -------- | ---------- | ---------------------- |
| 0.85          | 0.80            | 0.0001   | **125.60** | 3.00                   |
| 0.70          | 0.60            | 0.0000   | 125.49     | 5.00                   |
| 0.50          | 0.50            | 0.0000   | 122.90     | **10.00**              |
           |

write now this model from scratch is making too much mistakes is not sufficient for 7 class object detection task .Future work: Data should be augmented uniformly to handle class imbalance 