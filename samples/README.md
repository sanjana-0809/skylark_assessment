# Sample Predictions

This folder contains anonymized prediction visualizations from the validation set.

Each image is a three-panel figure:

| Panel | What it shows |
| --- | --- |
| **Left** | Full image with ground-truth marker (green circle) and predicted location (red ✕) |
| **Middle** | Zoomed-in crop around the marker with predicted vs actual shape classification |
| **Right** | Predicted heatmap overlaid on the image — visualizes what the model is "looking at" |

Filenames encode the marker shape and pixel error: `sample_NN_<Shape>_<error>px.png`.

Client and project identifying information has been stripped from filenames per professional practice.
