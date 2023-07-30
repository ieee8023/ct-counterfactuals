# 3D Densenet for Computed Tomography Based Diabetes Prediction (5 Year Incidence) - Hosted on Hugging Face at https://huggingface.co/louisblankemeier/ct_diabetes_prediction 

## Model Description

This model is a 3D Densenet trained to predict the incidence of diabetes within 5 years. The model was trained on a dataset of 5,629 full computed tomography (CT) scans, of which 829 were positive scans (i.e., those who develop diabetes in 5 years and no diagnosis at time of CT scan). The model achieves an AUROC (Area Under the Receiver Operating Characteristic) of approximately 0.7, suggesting that the model performs reasonably well in distinguishing between the two classes.

It's important to note that the labels used for training might be a bit noisy, as they were acquired from diabetes ICD codes.

## Model Details

- **Architecture**: 3D Densenet121 (https://docs.monai.io/en/stable/_modules/monai/networks/nets/densenet.html#DenseNet121)
- **Optimization**: Adam with 1e-5 learning rate and default parameters
- **Training set size**: 5,629 scans (829 positive)
- **Validation set size**: 1,875 scans (278 positive)
- **Test set size**: 1,912 scans (286 positive)
- **Training settings**: Trained with a batch size of 10 on a single 48 GB A6000 GPU for 10 epochs. We provide both the final checkpoint, and the best checkpoint on the validation set (trained for approximately 9 epochs).
- **Test set performance**: AUROC: 0.7335, AUPRC: 0.3111

## Dataset Details

- **Source**: Stanford Hospital Emergency Department
- **Type of Scans**: Abdomino-pelvic CT scans
- **Spacing**: 3mm in the axial dimension and 1.5mm in the in-plane dimensions
- **Orientation**: RAS+

## Required Preprocessing

- Resample the image to 1.5mm in the in-plane dimensions and 3mm in the axial dimension
- Ensure that the image is in the RAS+ orientation
- Divide the Hounsfield unit (HU) values by 1024
- Center crop to 250 x 250 in the in-plane dimensions and 150 in the axial dimension or pad with zeros equally on all sides if the image is smaller than (250, 250, 150)

## Intended Use

This model is intended for use in predicting the 5-year incidence of diabetes from CT scans. As with all medical predictive models, this should be used as a supplemental tool in conjunction with clinical judgement.

## Limitations

While this model has reasonable performance in predicting 5-year incidence of diabetes, it should be noted that the labels from the training data are a bit noisy due to being sourced from ICD codes. This could potentially limit the precision of the model. Furthermore, the model's performance metrics (AUROC and AUPRC) suggest that it may have limited sensitivity and specificity. Therefore, caution should be taken when interpreting its outputs. 

The model has also only been trained and tested on data from a single institution (Stanford Hospital), which might limit its generalizability to other populations and healthcare settings. It's recommended to validate the model's performance on external datasets before deploying it in other clinical settings.