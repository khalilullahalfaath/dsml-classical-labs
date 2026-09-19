# Customer Segmentation Service

> Clustering model to segment customers based on purchasing behavior and demographics, deployed as REST API

**Status:** Planning | **Duration:** 0 days | **Level:** Beginner

- [Customer Segmentation Service](#customer-segmentation-service)
  - [Project Brief](#project-brief)
  - [Learning Objectives](#learning-objectives)
  - [Tech Stack](#tech-stack)
  - [Dataset](#dataset)
  - [Project Structure](#project-structure)
  - [🚀 Getting Started](#-getting-started)
    - [1. Setup Environment](#1-setup-environment)
    - [2. Download Dataset](#2-download-dataset)
    - [3. Run EDA notebook](#3-run-eda-notebook)
    - [4. Run training script](#4-run-training-script)
    - [5. Start API Server](#5-start-api-server)
    - [6. Test API](#6-test-api)
  - [Core Requirements](#core-requirements)
  - [📋 Implementation Plan](#-implementation-plan)
  - [Fase 1 — Data \& EDA](#fase-1--data--eda)
  - [Fase 2 — RFM Feature Engineering](#fase-2--rfm-feature-engineering)
  - [Fase 3 — Clustering K-Means](#fase-3--clustering-k-means)
  - [Fase 4 — Clustering DBSCAN](#fase-4--clustering-dbscan)
  - [Fase 5 — Profiling \& Visualisasi](#fase-5--profiling--visualisasi)
  - [Fase 6 — API (FastAPI)](#fase-6--api-fastapi)
  - [Fase 7 — Testing \& Documentation](#fase-7--testing--documentation)
  - [Fase 8 — Frontend *(Opsional / Bonus)*](#fase-8--frontend-opsional--bonus)
  - [Fase 9 — Bonus](#fase-9--bonus)
  - [Urutan Pengerjaan](#urutan-pengerjaan)
  - [🔌 API Endpoints (Planned)](#-api-endpoints-planned)
    - [`POST /segment`](#post-segment)
    - [`GET /clusters`](#get-clusters)
    - [`GET /health`](#get-health)
    - [`GET /`](#get-)
  - [📈 Results](#-results)
    - [Model Comparison](#model-comparison)
    - [Cluster Profiles](#cluster-profiles)
    - [API Performance](#api-performance)
  - [📊 Model Card](#-model-card)
    - [Model Information](#model-information)
    - [Intended Use](#intended-use)
    - [Training Data](#training-data)
    - [Performance Metrics](#performance-metrics)
    - [Limitations](#limitations)
    - [Ethical Considerations](#ethical-considerations)
  - [Deliverables](#deliverables)
  - [Bonus Challenges](#bonus-challenges)
  - [📚 References](#-references)
  - [📝 Next Steps](#-next-steps)
  - [📧 Contact](#-contact)


## Project Brief

**The Request:**
"Client mau segmentasi customer berdasarkan perilaku belanja mereka. Biar bisa target marketing yang lebih tepat sasaran gitu."

**Objective:** Segment customers by purchasing behavior for targeted marketing.

**Success Criteria:**
- Clear, interpretable customer segments
- Silhouette score > 0.5 for clustering
- API response time < 200ms (P95)

**Out of scope:** 
- Real-time streaming segmentation
- CRM integration
- Individual customer lifetime value prediction
- Churn prediction (separate model recommended)

## Learning Objectives

- K-means and DBSCAN clustering algorithms
- RFM (Recency, Frequency, Monetary) feature engineering from raw transaction data
- Elbow method and silhouette score for optimal cluster selection
- Cluster profiling and business interpretation
- Deploying clustering models as REST API

## Tech Stack

- **Language:** Python 3.10+
- **ML Libraries:** scikit-learn (clustering), pandas (data manipulation), numpy
- **API Framework:** FastAPI (REST API)
- **Visualization:** matplotlib, seaborn (static plots), plotly (bonus: interactive)
- **Tools:** Jupyter Notebook, Postman, Git, joblib (model serialization)

## Dataset

**Primary:** [Online Retail (UCI)](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
- ~25,000 rows, 8 columns
- Features: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
- Transactional data from UK-based online retailer (2010-2011)

**Alternative:** [Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- Simpler dataset with pre-computed features
- Good for quick prototyping

## Project Structure

```
customer-segmentation-service/
├── data/
│   ├── raw/                    # Original dataset from Kaggle
│   │   └── online_retail.csv
│   └── processed/              # RFM features (to be generated)
│       └── rfm_features.csv
├── models/                     # Trained clustering models (.pkl)
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   └── cluster_profiles.json
├── notebooks/
│   ├── 01-eda.ipynb           # Exploratory Data Analysis
│   └── 02-clustering.ipynb    # Clustering experiments
├── src/
│   ├── preprocessing.py        # RFM feature engineering
│   ├── training.py            # Model training script
│   └── api.py                 # FastAPI application
├── tests/
│   └── test_api.py            # API unit tests
├── postman/
│   └── customer_segmentation.postman_collection.json
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### 1. Setup Environment

```bash
cd customer-segmentation-service
    
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
    
pip install -r requirements.txt
```

### 2. Download Dataset

Download [Online Retail Dataset](https://www.kaggle.com/datasets/carrie1/ecommerce-data) from Kaggle and place in `data/raw/online_retail.csv`

Or use Kaggle CLI:
```bash
kaggle datasets download -d carrie1/ecommerce-data
unzip ecommerce-data.zip -d data/raw/
mv data/raw/data.csv data/raw/online_retail.csv
```

### 3. Run EDA notebook

```bash
jupyter notebook notebooks/eda.ipynb
```

Explore:
- Transaction distributions
- Customer purchase patterns
- Missing values and data quality
- RFM feature engineering

### 4. Run training script

```bash
python src/training.py
```

This will:
- Load and preprocess data
- Engineer RFM features
- Train K-Means and DBSCAN models
- Save best model to `models/`
- Generate cluster profiles

### 5. Start API Server

```bash
uvicorn src.api:app --reload
```

API will be available at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### 6. Test API

Using cURL:
```bash
curl -X POST "http://localhost:8000/segment" \
  -H "Content-Type: application/json" \
  -d '{"recency": 30, "frequency": 10, "monetary": 1500.50}'
```

Or import Postman collection from `postman/customer_segmentation.postman_collection.json`

## Core Requirements

- [ ] RFM feature engineering from raw transaction records
- [ ] K-Means clustering with elbow method for optimal K
- [ ] DBSCAN comparison with parameter tuning
- [ ] Cluster profiling and visualization (scatter plots, radar charts)
- [ ] API endpoint to assign segment to a new customer based on their RFM values
- [ ] Silhouette score > 0.5 achieved
- [ ] API response time < 200ms (P95)

## 📋 Implementation Plan

## Fase 1 — Data & EDA

* [x] Download dataset Online Retail (UCI)
* [x] Load dataset ke Pandas
* [x] Cleaning: drop null `CustomerID`
* [x] Cleaning: remove cancelled orders (`InvoiceNo` starting `C`)
* [x] EDA: distribusi transaksi
* [x] EDA: revenue per customer
* [x] EDA: date range

## Fase 2 — RFM Feature Engineering

* [ ] Hitung **Recency**: `(max date - last purchase date)` per customer
* [ ] Hitung **Frequency**: jumlah invoice unik per customer
* [ ] Hitung **Monetary**: total spending per customer
* [ ] Analisis outlier dan skew
* [ ] Terapkan log transform jika diperlukan
* [ ] Scaling menggunakan `StandardScaler`

## Fase 3 — Clustering K-Means

* [ ] Implementasi K-Means
* [ ] Jalankan Elbow Method (`inertia` vs `K`)
* [ ] Tentukan kandidat `K` optimal
* [ ] Validasi menggunakan Silhouette Score
* [ ] Pastikan Silhouette Score > 0.3
* [ ] Fit final K-Means model
* [ ] Assign cluster label ke setiap customer
* [ ] Simpan model K-Means

## Fase 4 — Clustering DBSCAN

* [ ] Implementasi DBSCAN
* [ ] Buat k-distance graph
* [ ] Tuning `eps`
* [ ] Tuning `min_samples`
* [ ] Hitung jumlah cluster
* [ ] Hitung jumlah noise points
* [ ] Hitung Silhouette Score
* [ ] Bandingkan DBSCAN vs K-Means
* [ ] Pilih model final untuk deployment

## Fase 5 — Profiling & Visualisasi

* [ ] Buat cluster profiling
* [ ] Hitung rata-rata RFM per cluster
* [ ] Interpretasikan karakteristik setiap cluster
* [ ] Berikan label bisnis pada setiap segment
* [ ] Contoh: `Champions`, `At Risk`, dll.
* [ ] Buat scatter plot RFM 2D per cluster
* [ ] Buat scatter plot RFM 3D per cluster
* [ ] Buat radar chart per segment

## Fase 6 — API (FastAPI)

* [ ] Buat endpoint `POST /predict-segment`
* [ ] Tentukan schema input RFM raw customer
* [ ] Load scaler menggunakan `pickle`/`joblib`
* [ ] Load model clustering
* [ ] Implementasi preprocessing input
* [ ] Implementasi prediksi segment
* [ ] Return segment label
* [ ] Return probability/distance jika tersedia
* [ ] Buat response schema
* [ ] Test API secara lokal

## Fase 7 — Testing & Documentation

* [ ] Buat Postman collection
* [ ] Test endpoint dengan beberapa customer case
* [ ] Test valid input
* [ ] Test invalid input
* [ ] Test edge cases
* [ ] Dokumentasikan cara menjalankan project
* [ ] Dokumentasikan struktur project
* [ ] Dokumentasikan endpoint API
* [ ] Dokumentasikan interpretasi setiap segment
* [ ] Dokumentasikan cara menggunakan API

## Fase 8 — Frontend *(Opsional / Bonus)*

* [ ] Buat Streamlit app
* [ ] Buat form input RFM
* [ ] Connect Streamlit ke FastAPI
* [ ] Tampilkan hasil segment
* [ ] Tampilkan karakteristik segment
* [ ] Tambahkan visualisasi sederhana

## Fase 9 — Bonus

* [ ] Buat interactive visualization menggunakan Plotly/Bokeh
* [ ] Implementasi automatic K selection berdasarkan Silhouette Score
* [ ] Bandingkan beberapa nilai `K` secara otomatis
* [ ] Implementasi migration tracking
* [ ] Compare segment customer antar periode
* [ ] Visualisasikan perpindahan segment

## Urutan Pengerjaan

* [ ] **Fase 1 — Data & EDA**
* [ ] **Fase 2 — RFM Feature Engineering**
* [ ] **Fase 3 — K-Means** *(bisa paralel dengan Fase 4)*
* [ ] **Fase 4 — DBSCAN** *(bisa paralel dengan Fase 3)*
* [ ] **Fase 5 — Profiling & Visualisasi**
* [ ] **Fase 6 — FastAPI**
* [ ] **Fase 7 — Testing & Documentation**
* [ ] **Fase 8 — Streamlit (opsional)**
* [ ] **Fase 9 — Bonus**


## 🔌 API Endpoints (Planned)

### `POST /segment`
Assign a customer to a segment based on RFM values.

**Request Body:**
```json
{
  "recency": 30,        
  "frequency": 10,      
  "monetary": 1500.50   
}
```

**Response:**
```json
{
  "cluster_id": 1,
  "cluster_name": "At-Risk Customers",
  "cluster_description": "Customers with declining activity",
  "rfm_scores": {
    "recency": 30,
    "frequency": 10,
    "monetary": 1500.50
  },
  "recommendations": [
    "Send re-engagement email campaign",
    "Offer 15% loyalty discount",
    "Highlight new product arrivals"
  ]
}
```

### `GET /clusters`
Get all cluster profiles and statistics.

**Response:**
```json
{
  "total_clusters": 4,
  "model_info": {
    "algorithm": "K-Means",
    "silhouette_score": 0.62,
    "trained_date": "2026-05-08"
  },
  "clusters": [
    {
      "cluster_id": 0,
      "cluster_name": "High-Value Customers",
      "description": "Frequent buyers with high spending",
      "size": 3750,
      "percentage": 15.0,
      "avg_rfm": {
        "recency": 15,
        "frequency": 25,
        "monetary": 5000
      },
      "characteristics": [
        "Recent purchases (< 30 days)",
        "High purchase frequency (> 20 orders)",
        "High total spending (> $3000)"
      ]
    }
  ]
}
```

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-05-08T10:30:00"
}
```

### `GET /`
Root endpoint with API information.

## 📈 Results

> **Note:** This section will be filled after model training is complete.

### Model Comparison

_To be updated after training K-Means and DBSCAN models._

**Expected format:**

| Model | Silhouette Score | Davies-Bouldin | Optimal K/eps | Training Time |
|-------|------------------|----------------|---------------|---------------|
| K-Means | TBD | TBD | K=? | TBD |
| DBSCAN | TBD | TBD | eps=?, min_samples=? | TBD |

**Best Model:** _To be determined_

### Cluster Profiles

_To be updated with cluster characteristics and business interpretation._

**Expected cluster types:**
- **High-Value Customers:** Frequent, recent, high spending
- **At-Risk Customers:** Declining activity, medium value
- **New Customers:** Recent join, low activity, growth potential
- **Lost Customers:** No recent activity, low engagement

### API Performance

_To be benchmarked after API deployment._

**Target metrics:**
- Average Response Time: < 100ms
- P95 Latency: < 200ms
- P99 Latency: < 500ms
- Throughput: > 100 requests/second

## 📊 Model Card

### Model Information

- **Algorithm:** K-Means Clustering
- **Alternative:** DBSCAN (for comparison)
- **Version:** 1.0.0
- **Training Date:** _To be updated_
- **Framework:** scikit-learn
- **Model Size:** _To be measured_

### Intended Use

**Primary Use Case:**
- Segment customers into distinct groups based on purchasing behavior (RFM analysis)
- Enable targeted marketing campaigns per segment
- Identify high-value and at-risk customers for retention strategies
- Inform product recommendations and personalization

**Users:**
- Marketing teams for campaign planning and budget allocation
- Product teams for feature prioritization
- Customer success teams for retention and engagement strategies
- Business analysts for customer insights

**Out-of-Scope:**
- Real-time streaming segmentation (batch processing only)
- Individual customer lifetime value prediction (use separate regression model)
- Churn prediction (use separate classification model)
- Product recommendation (use collaborative filtering)
- CRM system integration (API only)

### Training Data

**Dataset:** Online Retail Dataset (UCI)
- **Source:** [Kaggle - Online Retail](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
- **Size:** ~25,000 transactions, ~4,000 unique customers
- **Original Features:** InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
- **Engineered Features:** RFM (Recency, Frequency, Monetary)
- **Time Period:** December 2010 - December 2011
- **Geography:** Primarily UK-based online retailer

**Feature Engineering:**
- **Recency (R):** Days since last purchase (lower = more recent)
- **Frequency (F):** Total number of transactions per customer
- **Monetary (M):** Total spending per customer (sum of Quantity × UnitPrice)

**Preprocessing Steps:**
1. Remove cancelled transactions (negative quantities)
2. Filter out non-customer transactions (missing CustomerID)
3. Remove outliers (e.g., wholesale orders with extreme quantities)
4. Calculate RFM metrics per customer
5. Standardize features using StandardScaler (mean=0, std=1)

**Data Splits:**
- No train/test split (unsupervised learning)
- Full dataset used for clustering
- Validation via silhouette score and business interpretation

### Performance Metrics

> **Note:** To be measured after training.

**Target Clustering Quality Metrics:**
- **Silhouette Score:** > 0.5 (good cluster separation)
  - Range: [-1, 1], higher is better
  - Measures how similar objects are to their own cluster vs other clusters
- **Davies-Bouldin Index:** < 1.0 (lower is better)
  - Measures average similarity between clusters
- **Calinski-Harabasz Score:** Higher is better
  - Ratio of between-cluster to within-cluster variance

**Expected Cluster Distribution:**
- Aim for balanced clusters (no cluster < 5% of data)
- 3-5 clusters expected (to be determined by elbow method)
- Each cluster should have clear business interpretation

**Cluster Stability:**
- Re-run clustering with different random seeds
- Measure consistency of cluster assignments
- Target: > 90% assignment stability

### Limitations

**Model Limitations:**
- **Static segmentation:** Customers don't automatically migrate between segments over time (requires re-training)
- **RFM-only:** Does not account for product preferences, browsing behavior, or demographic data
- **K-Means assumptions:** Assumes spherical clusters of similar size (may miss complex patterns)
- **Outlier sensitivity:** Very high spenders or bulk buyers may skew cluster centroids
- **Historical bias:** Model trained on past data may not reflect current customer trends

**Data Limitations:**
- **Geographic bias:** Dataset primarily from UK customers (may not generalize globally)
- **Temporal bias:** Data from 2010-2011 (purchasing patterns may have changed)
- **Seasonal effects:** Dataset covers only 1 year (seasonal patterns not fully captured)
- **B2B vs B2C:** Dataset may contain both business and consumer customers (not distinguished)
- **Missing context:** No demographic data (age, gender, location) or product categories

**Technical Limitations:**
- **Batch processing only:** Not suitable for real-time segmentation
- **Manual interpretation:** Cluster business meaning requires human analysis
- **Scalability:** K-Means performance degrades with very large datasets (> 1M customers)
- **Cold start:** New customers with no purchase history cannot be segmented

**When NOT to Use This Model:**
- For individual customer lifetime value prediction (use regression)
- For churn prediction (use classification)
- For product recommendations (use collaborative filtering)
- When real-time segmentation is required (use streaming ML)

### Ethical Considerations

**Potential Biases:**
- **Economic bias:** May disadvantage lower-income customer segments
  - Risk: "Lost customers" segment may be unfairly deprioritized
  - Mitigation: Ensure all segments receive baseline service quality
- **Geographic bias:** Dataset over-represents UK customers
  - Risk: Model may not generalize to other markets
  - Mitigation: Re-train model with local data when expanding to new regions
- **Temporal bias:** Purchasing patterns may be seasonal or trend-driven
  - Risk: Segments may not be stable over time
  - Mitigation: Re-train model quarterly or when business conditions change

**Responsible Use:**
- **Segments should inform, not dictate:** Use segmentation as one input for marketing decisions, not the sole factor
- **Avoid discriminatory pricing:** Do not use segments to charge different prices for the same product
- **Transparency:** Clearly communicate to customers how their data is used for segmentation
- **Opt-out mechanisms:** Provide customers with ability to opt out of targeted marketing
- **Regular audits:** Monitor segment distributions for unexpected shifts or biases

**Privacy & Compliance:**
- **Data anonymization:** Remove personally identifiable information (PII) before training
- **GDPR compliance:** Ensure right to erasure (delete customer data on request)
- **CCPA compliance:** Provide transparency about data usage and opt-out options
- **Data retention:** Define and enforce data retention policies (e.g., delete data after 2 years)
- **Access controls:** Limit access to customer segments to authorized personnel only

**Fairness Considerations:**
- **Equal opportunity:** Ensure all customer segments have access to promotions and support
- **Avoid reinforcing inequality:** Don't use segmentation to exclude disadvantaged groups
- **Monitor outcomes:** Track conversion rates and satisfaction across segments to detect disparities
- **Human oversight:** Marketing decisions based on segments should be reviewed by humans

## Deliverables

- [ ] Jupyter notebook with EDA and clustering analysis
- [ ] Trained clustering model (K-Means) with scaler (.pkl files)
- [ ] REST API with segment assignment endpoint (FastAPI)
- [ ] Postman collection for API testing
- [ ] README with cluster profiles and business interpretation
- [ ] Model Card with performance metrics and limitations
- [ ] Unit tests for API endpoints
- [ ] Performance benchmark report (response time, P95)

## Bonus Challenges

- [ ] Interactive cluster visualization (Plotly/Bokeh)
- [ ] Automatic optimal K selection using silhouette analysis
- [ ] Customer migration tracking between segments over time
- [ ] Build a simple frontend (Streamlit/HTML) to interact with the API
- [ ] Dockerize the application for easy deployment
- [ ] Add batch prediction endpoint (segment multiple customers at once)
- [ ] Implement caching for faster repeated predictions
- [ ] Add explainability: show which RFM features contributed most to segment assignment

## 📚 References

- [RFM Analysis Guide](https://www.putler.com/rfm-analysis/)
- [K-Means Clustering - scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [DBSCAN - scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
- [Customer Segmentation Best Practices](https://www.optimove.com/resources/learning-center/customer-segmentation)

## 📝 Next Steps

After completing core requirements:

1. **Model Improvement:**
   - Experiment with hierarchical clustering
   - Try Gaussian Mixture Models (GMM) for soft clustering
   - Add more features (product categories, time-based patterns)

2. **Production Readiness:**
   - Add authentication (API keys)
   - Implement rate limiting
   - Set up monitoring and logging
   - Create CI/CD pipeline

3. **Business Integration:**
   - Create marketing playbooks per segment
   - Set up automated email campaigns
   - Build dashboard for business users
   - Integrate with CRM system

4. **Advanced Features:**
   - Customer lifetime value prediction per segment
   - Churn prediction model
   - Product recommendation engine
   - A/B testing framework for marketing campaigns

## 📧 Contact

Questions or feedback? Open an issue or reach out!

---

**Project Timeline:** May 8 - TBD | **Total Hours:** TBD hours | **Status:** Planning Phase