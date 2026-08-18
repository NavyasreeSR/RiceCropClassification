# Rice Crop Classification from Sentinel Satellite Data

**EY Open Science Data Challenge 2023 — Level 1**
Indiana University · Navya Sree Santhapeta (`nsantha@iu.edu`)

A binary classifier that distinguishes **rice** from **non-rice** fields at 250 geo-locations
in the **An Giang province of Vietnam**, built on Sentinel-1 radar and Sentinel-2 optical
imagery accessed through the **Microsoft Planetary Computer**.

The submitted model reached an **F1 score of 0.85**, up from the challenge benchmark's **0.57**.

## Background

Accurately mapping where rice — a staple food crop for much of the world — is grown supports
food-security planning and agricultural resource allocation. The EY Open Science Data
Challenge provided labelled geo-locations in Vietnam for 2020 and a benchmark logistic
regression notebook scoring F1 = 0.57; the task was to improve on it.

The core improvements in this work over the benchmark:

- **Full-year signal instead of a single date.** The benchmark extracts VV/VH for one
  two-day window (`2020-03-20/2020-03-21`). This model extracts values across the entire year
  and averages them, so different land classes are separated by their annual signature rather
  than a single snapshot.
- **Spatial windowing.** Rather than sampling one pixel, band values are aggregated over
  3×3, 5×5, and 7×7 pixel boxes around each coordinate and averaged, reducing per-pixel noise.
- **Higher-correlation derived features.** RVI (Radar Vegetation Index) and NDVI are computed
  from the raw bands and used as predictors, as they correlate more strongly with rice presence
  than raw VV/VH.
- **Tuned logistic regression.** `C`, `penalty`, `solver`, and `class_weight` were refined
  rather than left at benchmark defaults.

## Repository contents

| File | Description |
|---|---|
| `Crop_identification.ipynb` | Main notebook (73 cells). Full pipeline: load labels → query Sentinel-1 RTC via Planetary Computer STAC → build features → scale → train logistic regression → evaluate → write submission. |
| `crop_identification_Sentinel 1 Phenology.ipynb` | Sentinel-1 exploration. Computes the Radar Vegetation Index time series over a known rice field and plots vegetation phenology. Radar penetrates cloud, so this signal is available year-round. |
| `crop_identification_Sentinel 2 cloud filtering.ipynb` | Sentinel-2 exploration. Uses the SCL band to mask no-data, saturated, cloud, cloud-shadow, and water pixels, then compares filtered vs. unfiltered mean NDVI time series. |
| `def get_sentinel_data().txt` | Standalone copy of the modified `get_sentinel_data()` function — the multi-window (3×3 / 5×5 / 7×7) RVI extractor that replaces the benchmark's single-pixel version. |
| `Crop_Location_Data_20221201.csv` | Training labels: 600 coordinate pairs tagged `Rice` / `Non Rice`. |
| `Crop_Location_Data_20221201(1).csv` | Same 600 rows with the computed `Mean_RVI` feature appended. |
| `challenge_1_submission_rice_crop_prediction.csv` | Final submission: predictions for the 250 held-out test coordinates. |
| `Crop_identification.tex` | LaTeX source of the project report. |
| `Crop_identification_report.txt` | Plain-text version of the report — abstract, related work, methodology, results, conclusion, references. |

## Data

**Labels.** 600 `(latitude, longitude)` pairs in An Giang province, Vietnam, each tagged
`Rice` or `Non Rice`, for the year 2020. Test set: 250 unlabelled coordinates.

**Imagery.** Queried live from the
[Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) STAC API:

- **`sentinel-1-rtc`** — radiometrically terrain-corrected radar. Bands `vv`, `vh`, loaded at
  10 m/pixel. Cloud-independent, so usable at any date.
- **`sentinel-2-l2a`** — optical. Bands `red`, `green`, `blue`, `nir`, `SCL`, loaded at
  20 m/pixel. Used for NDVI and cloud-mask exploration.

### Feature definitions

**RVI (Radar Vegetation Index)** from Sentinel-1, where `DOP = VV / (VV + VH)`:

```
RVI = sqrt(DOP) × (4 × VH) / (VV + VH)
```

**NDVI (Normalized Difference Vegetation Index)** from Sentinel-2:

```
NDVI = (NIR − Red) / (NIR + Red)
```

NDVI ranges 0.0–1.0: low values (0.0–0.25) indicate bare soil or water, middle values
(0.25–0.6) crops in growth, high values (0.6–1.0) peak vegetation.

## Method

1. **Data collection** — for each labelled coordinate, query Sentinel-1 RTC over the full year,
   load `vv`/`vh` into an xarray via Open Data Cube, and average across three bounding boxes
   (0.0003°, 0.0004°, 0.0007° ≈ 3×3, 5×5, 7×7 pixels). Compute mean RVI per box, then average.
2. **Preprocessing** — normalize features with scikit-learn's `StandardScaler`.
3. **Split** — 70% train / 30% test via `train_test_split`, stratified on the label.
4. **Train** — binary `LogisticRegression`, with `C`, `penalty`, `solver`, and `class_weight`
   tuned against the benchmark's defaults.
5. **Evaluate** — classification report and confusion matrix on both the training set
   (in-sample) and the held-out set (out-of-sample), to check for overfitting.
6. **Submit** — apply the fitted scaler and model to the 250 test coordinates and write
   `challenge_1_submission_rice_crop_prediction.csv`.

## Results

| Model | F1 score |
|---|---|
| Challenge benchmark (single-date VV/VH, default logistic regression) | 0.57 |
| **This model** (full-year multi-window RVI/NDVI, tuned logistic regression) | **0.85** |

The report's results section quotes 0.86 for the out-of-sample evaluation while the abstract
and conclusion state 0.85; 0.85 is the figure used throughout the writeup.

Despite the improvement over the benchmark, the submission did not advance to Level 2 of
the competition.

## Running the notebooks

A Planetary Computer subscription key is required to query the STAC API.

```bash
pip install planetary-computer pystac-client odc-stac rioxarray xarray dask \
            numpy pandas scikit-learn matplotlib seaborn ipyleaflet tqdm xrspatial
```

Set the key before running:

```python
import planetary_computer as pc
pc.settings.set_subscription_key("<YOUR_KEY>")
# or export PC_SDK_SUBSCRIPTION_KEY=<YOUR_KEY>
```

Then open `Crop_identification.ipynb` and run in order.

**Notes:**

- The notebooks contain leftover `pip install` cells pinning old versions
  (`numpy==1.16.6`, `dask==2.30.0`). Skip these — install with current versions instead.
- Cell 63 reads `challenge_1_submission_template.csv`, which is not in this repo. Substitute
  the coordinate list from `challenge_1_submission_rice_crop_prediction.csv` to reproduce the
  submission step.
- Full-year extraction across 600 coordinates issues a large number of STAC queries and
  raster loads; expect the feature-extraction cells to run for a long time.

## Future work

- Explore alternative classifiers — SVMs, random forests — and additional derived features.
- Bring in domain expertise to add contextual features and validate the labels.
- Extend the geographic scope beyond An Giang to test how well the approach generalizes.

## References

1. Lobell, D.B. *"The use of satellite data for crop yield gap analysis."* Field Crops Research, 2013, 143, 56–64.
2. Hao, P.; Wang, L.; Zhan, Y.; Niu, Z. *"Using moderate-resolution temporal NDVI profiles for high-resolution crop mapping in years of absent ground reference data: a case study of Bole and Manas Counties in Xinjiang, China."* ISPRS Int. J. Geo-Inf., 2016, 5, 67.
3. Löw, F.; Michel, U.; Dech, S.; Conrad, C. *"Impact of feature selection on the accuracy and spatial uncertainty of per-field crop classification using support vector machines."* ISPRS J. Photogramm. Remote Sens., 2013, 85, 102–119.

---

*Academic coursework / competition submission, EY Open Science Data Challenge 2023.*
