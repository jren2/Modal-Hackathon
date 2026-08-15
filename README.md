# EgoTrim · Behavioral-diversity curation for EgoVerse

EgoTrim is both a data pipeline and interactive demo for inspecting how much
EgoVerse footage can be removed while retaining the behaviors that matter. The scope of the project is only on folding clothes data but with the intent to scale up to the entire EgoVerse dataset. The current app is hosted at https://jren2--egoverse-segment-browser-web.modal.run/ with a more in depth description of the pipeline below.

## Methodology

<img width="754" height="484" alt="image" src="https://github.com/user-attachments/assets/4612ce62-0086-4ebc-bd4a-141d78b10c9a" />

1. Data Ingestion
3. Clip Segmentation
4. Feature Engineering
5. Weight Training
6. Clustering
