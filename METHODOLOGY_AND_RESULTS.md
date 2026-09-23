# Methodology

## Study design and image data

The project developed an image-analysis workflow for identifying four common
microplastic polymers—polyethylene terephthalate (PET), polypropylene (PP),
polystyrene (PS), and polyvinyl chloride (PVC)—from ultraviolet (UV)
fluorescence images. Two complementary pipelines were implemented. The first
is an image-level baseline classifier that assigns one polymer label to an
image. The second is the principal multi-class detection pipeline, which
localizes and classifies individual particles in mixed images and estimates
their physical dimensions.

The image-level dataset was arranged into training, validation, and test
subsets. Each subset contained 25 images per polymer for validation, 15 images
per polymer for testing, and 125 images per polymer for training, giving 500
training images, 100 validation images, and 60 test images in total. The
object-detection dataset contained 636 training images, 120 validation images,
and 77 test images. The four object classes were encoded as PET, PP, PS, and
PVC. The test split contained 2,219 annotated particle instances. Images were
exported without Roboflow preprocessing or colour augmentation because polymer
identity was represented partly by fluorescence colour.

## Image-level baseline pipeline

Images were first read in BGR format and the circular dish/filter region was
identified using the Hough circle transform. When a circle could not be
detected, a centred fallback circle covering approximately 84% of the shorter
image dimension was used. The circular region was cropped, resized to
512 × 512 pixels, and represented with a binary circular mask.

Particle pixels were segmented in HSV colour space using three complementary
rules: pixels outside the blue-background hue range (OpenCV hue values 100–128),
bright pixels with reduced saturation (saturation <150 and value >200), and
very bright pixels (value >230). The resulting mask was restricted to the
circular region, cleaned with a 3 × 3 elliptical morphological opening, and
connected components smaller than 8 pixels² were removed.

An 88-dimensional feature vector was calculated for each image. It consisted of
normalized HSV histograms (34 features), particle-pixel colour statistics (15),
global circular-region statistics (17), and aggregated particle morphology
features (22), including area, perimeter, circularity, aspect ratio, solidity,
and extent. Features were standardized with `StandardScaler` and classified
with an RBF-kernel support-vector machine (SVM; C = 50, gamma = `scale`). The
saved baseline model was trained using the training and validation data, and
was evaluated on the independent test subset.

## Multi-class particle detection

The principal pipeline used a pretrained YOLO11s object-detection model. The
model was trained for 100 epochs with 1,024-pixel input images and a batch size
of 8 on an Apple Metal Performance Shaders (MPS) device. Training used the
four annotated polymer classes and retained geometric augmentation, including
translation, scaling, horizontal flipping, and mosaic augmentation. Hue,
saturation, and value augmentation were disabled (`hsv_h = hsv_s = hsv_v = 0`)
to preserve the fluorescence colour signal used for polymer discrimination.
The best checkpoint from the third training run was selected using validation
performance. Detection was evaluated at the configured confidence threshold of
0.25 and an input size of 1,024 pixels.

## Physical size estimation

For each detected image, the circular filter paper was detected independently
with the Hough circle transform. Its known diameter of 47 mm was used for
image-specific calibration:

\[
\text{scale} = \frac{47,000\ \mu\text{m}}{2r},
\]

where \(r\) is the fitted filter radius in pixels. If the filter circle could
not be detected, a fallback scale of 39.8 µm/pixel was used and the result was
flagged as approximate.

Each YOLO bounding box was refined using local background subtraction in CIELAB
colour space. Otsu thresholding and morphological closing were then used to
obtain the particle mask when sufficient local contrast was present. A rotated
minimum-area rectangle fitted to the particle contour provided the major and
minor axes. If the mask contained fewer than six pixels or could not be
resolved, the YOLO box was used instead. Equivalent circular diameter (ECD) was
calculated from the estimated particle area:

\[
\text{ECD} = 2\sqrt{\frac{A}{\pi}},
\]

and pixel measurements were converted to micrometres using the image-specific
calibration scale.

# Results

## Image-level polymer classification

The saved SVM classified all images correctly in the available evaluation
subsets. Training accuracy was 100% (500/500), validation accuracy was 100%
(100/100), and test accuracy was 100% (60/60). On the test set, each class had
15 images and achieved precision, recall, and F1-score of 1.00. The test
confusion matrix contained no misclassifications:

| Actual \ Predicted | PET | PP | PS | PVC |
|---|---:|---:|---:|---:|
| PET | 15 | 0 | 0 | 0 |
| PP | 0 | 15 | 0 | 0 |
| PS | 0 | 0 | 15 | 0 |
| PVC | 0 | 0 | 0 | 15 |

These results indicate that the fluorescence and morphology features were
highly separable for the controlled, single-label image dataset. Because the
dataset is relatively small and the test images appear to come from the same
controlled imaging setup, the result should be interpreted as performance on
this dataset rather than as evidence of perfect generalization to new
instruments, lighting conditions, or environmental samples.

## Multi-class particle detection

The final YOLO11s run reached a best validation mAP@0.50–0.95 of 0.703 at
epoch 98, with validation mAP@0.50 of 0.969. On the held-out test split, the
model processed 77 images containing 2,219 annotated particles and achieved:

| Metric | Test result |
|---|---:|
| Precision | 0.939 |
| Recall | 0.954 |
| mAP@0.50 | 0.951 |
| mAP@0.50–0.95 | 0.670 |

The class-wise mAP@0.50–0.95 values were 0.675 for PET, 0.733 for PP, 0.766
for PS, and 0.506 for PVC. Thus, PS and PP were localized most accurately,
whereas PVC was the most difficult class under the stricter IoU range. The
overall recall shows that most annotated particles were detected, while the
lower mAP@0.50–0.95 relative to mAP@0.50 indicates that localization becomes
less precise under stricter overlap requirements.

## Measurement output

The application produces a per-particle record containing polymer identity,
detection confidence, centre coordinates, bounding-box dimensions, major and
minor axes, ECD, estimated area, aspect ratio, and the measurement method
(`mask` or `bbox`). It also provides counts and ECD summaries by polymer for
each analyzed image. No consolidated field-sample measurement table or
aggregate polymer-size distribution was included in the project artifacts;
therefore, the results above report model performance and describe the
measurement output rather than claiming an environmental abundance or size
distribution that was not directly recorded.

