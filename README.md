# Amazon ML Challenge 2025 – Smart Product Pricing Solution

# Team Optimizers

Team Members
1. Sahil Burnwal (Team Lead)
2. Sudhanshu Anand
3. Gagan C
4. Aditya Mohan Singh

Competition Platform: Unstop
Submission Date: October 13, 2025
Rank: Under 1000 among 82000 teams 

Problem Statement

Predict product prices accurately using multimodal data (text + images) for Amazon-like product listings.
The evaluation metric is SMAPE, emphasizing robustness across low-, medium-, and high-priced products.


Executive Summary

We propose a multimodal deep learning solution that fuses textual, visual, and engineered features to predict product prices.
Our final system uses:
1. TF-IDF + SVD for structured text understanding
2. ResNet50 image embeddings + PCA for visual signals
3. A 5-layer feed-forward neural network
4. A custom SMAPE-optimized hybrid loss function

Final Result

1. Cross-Validation SMAPE: 48.40
2. Relative improvement: ~11% over baseline
3. Low fold variance: ±0.30 → stable generalization

Methodology Overview
Key Data Observations

1. Price range: $0.13 – $2,796 → log transformation required
2. ~38% products priced below $10 → needs weighted learning
3. Pack size & quantity strongly affect price
4. Premium keywords (organic, imported, artisan) matter
5. Category-based price tiers exist (wine, cheese vs water, powder)

System Architecture

Input Features (220)
│
├── Text Features (115)
│   ├── Name TF-IDF + SVD (42)
│   ├── Bullet TF-IDF + SVD (55)
│   └── Char n-grams + SVD (18)
│
├── Engineered Features (34)
│   ├── Size / Pack / Volume (12)
│   ├── Premium Indicators (6)
│   ├── Category & Brand Encodings (2)
│   ├── Text Statistics (7)
│   └── Interaction Terms (7)
│
├── Image Features (71)
│   ├── ResNet50 → PCA (65)
│   └── Image-Text Interactions (6)
│
▼
BatchNorm → Dropout
▼
FC Layers (384 → 256 → 128 → 64 → 32)
▼
Output: Price Prediction


Feature Engineering
1️⃣ Size & Quantity Features (Most Important)
a. Unit-normalized size (oz)
b. Pack count detection
c. Total volume (size × pack)
d. Log & sqrt transformations
e. Binary flags (has_size, is_bulk)

2️⃣ Premium & Quality Indicators
a. Ultra-premium: imported, artisan, handcrafted
b. Premium: organic, gourmet
c. Quality: natural, authentic
d. Weighted premium score

3️⃣ Category & Brand Encoding
a. Rule-based category detection
b. Bayesian target encoding (m=10 smoothing)

4️⃣ Interaction Features
a. size × premium score
b. pack count × premium
c. brand × category
d. image × text embeddings


Image Processing Pipeline
1. Image download with retry & backoff
2. Resize → 224×224
3. Pretrained ResNet50 (no classifier)
4. PCA: 2048 → 65 dimensions
5. Missing/broken images → zero vector fallback

Model Training
🔧 Optimization Setup
1. Optimizer: AdamW
2. Learning Rate: 0.0008
3. Weight Decay: 0.0075
4. Batch Size: 384
5. Epochs: up to 200
6. Early Stopping: patience = 30
7. Scheduler: CosineAnnealingWarmRestarts
8. Target Transform: log1p(price)


Custom Loss Function (Core Innovation)
Total Loss =
0.20 × Weighted MSE
+ 0.20 × Huber Loss
+ 0.60 × SMAPE Loss

Why?
1. Directly optimizes competition metric
2. Handles outliers
3. Emphasizes low-price accuracy

Model Performance
Overall Metrics
Metric	Value
SMAPE	  48.40
RMSE	  $26.61
MAE	    $10.85
R²	    0.364
MAPE	  66.15%

Price-Range Breakdown
PriceRange	  SMAPE	  MAE	  % Data
$0–$10	      51.60	  $4.26	  38%
$10–$20	      33.98	  $4.82	  26%
$20–$50	      50.36	  $12.25	  25%
>$50	        67.04	  $45.17	  11%

Ablation Study
Removed Component	    SMAPE
Full Model	          48.40
– Size Features	      53.50
– Target Encoding	    52.60
– Image Features	    52.20
– Custom Loss	        51.10
– Premium Features	  50.50

Key Insight:
Size & pack information is the strongest price signal.


Challenges Faced
1. ❌ Transformer models rejected (API dependency)
2. ❌ Regex-only extraction failed on inconsistent formats
3. ⚠️ Image CDN throttling
4. ⚠️ Category imbalance & rare expensive items

Solutions
1. TF-IDF + SVD over transformers
2. Manual rule-based extraction
3. PCA + batch processing
4. Bayesian smoothing & heavy regularization

Future Improvements
1. Attention-based feature weighting
2. Price-bucketed models
3. Category-specific experts
4. Vision Transformers (ViT)
5. Multi-task learning (price + category)
6. Graph-based brand/category modeling

Conclusion
This project demonstrates that careful feature engineering + metric-aware optimization can outperform heavier models.
Our multimodal DNN achieves stable, competitive performance while remaining fully offline, reproducible, and efficient.


Note: The dataset used in this project is proprietary and governed by Amazon ML Challenge usage policies. As a result, raw data and derived features are not included in this repository. Only the complete training and feature engineering notebooks are shared to demonstrate methodology and implementation.
