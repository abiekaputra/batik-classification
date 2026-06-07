# Batik Pattern Classifier — Indonesian Cultural Heritage AI

A deep learning project that classifies traditional Indonesian batik fabric patterns using **MobileNetV2 transfer learning**. Batik is recognized by UNESCO as an **Intangible Cultural Heritage of Humanity**, and this project aims to leverage modern computer vision to help preserve, digitize, and identify these intricate patterns at scale.

---

## Highlights

- **Transfer Learning** with MobileNetV2 pre-trained on ImageNet
- **Fine-tuning** top layers for domain-specific adaptation
- **Data Augmentation** (rotation, flip, shift) to improve generalization
- **Streamlit web app** for interactive inference with confidence visualization
- **Multiple batik classes** representing distinct regional traditions
- Full training pipeline in a single Jupyter notebook

---

## Tech Stack

| Component       | Technology                          |
|----------------|-------------------------------------|
| Language        | Python 3.9+                         |
| Deep Learning   | TensorFlow / Keras                  |
| Base Model      | MobileNetV2 (ImageNet weights)      |
| Data Processing | NumPy, Pandas, PIL                  |
| Visualization   | Matplotlib, Seaborn, Plotly         |
| Web App         | Streamlit                           |
| Dataset         | Kaggle — `aldian/batik-kaggle`      |

---

## Prerequisites

### 1. Python Environment

```bash
pip install -r requirements.txt
```

### 2. Kaggle API Setup

The notebook downloads the dataset automatically via the Kaggle API.

1. Go to [https://www.kaggle.com/settings/account](https://www.kaggle.com/settings/account)
2. Scroll to **API** section → click **Create New Token**
3. A file `kaggle.json` will be downloaded
4. Place it at `~/.kaggle/kaggle.json`
5. Set permissions:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

---

## Getting Started

### Step 1 — Clone the Repository

```bash
git clone https://github.com/abiekaputra/batik-classification.git
cd batik-classification
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Download Dataset & Train Model

Open and run `notebook.ipynb` from top to bottom. It will:
- Download the dataset from Kaggle automatically
- Explore and visualize the data
- Train MobileNetV2 with transfer learning
- Fine-tune on top layers
- Save the model to `models/batik_model.h5`

**Manual dataset download (alternative):**

If you prefer to download manually:
1. Visit [https://www.kaggle.com/datasets/aldian/batik-kaggle](https://www.kaggle.com/datasets/aldian/batik-kaggle)
2. Download and extract the zip
3. Place images in this structure:
   ```
   data/batik/
     ├── parang/
     │   ├── image1.jpg
     │   └── ...
     ├── kawung/
     ├── megamendung/
     └── ...
   ```

### Step 4 — Run the Streamlit App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`, upload a batik image, and see the classification result with confidence scores.

---

## Project Structure

```
batik-classification/
├── notebook.ipynb          # Full training pipeline (CNN + fine-tuning)
├── app.py                  # Streamlit inference web app
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore
├── data/
│   ├── .gitkeep
│   └── batik/              # Dataset (not tracked by git)
│       ├── parang/
│       ├── kawung/
│       ├── megamendung/
│       └── ...
└── models/
    ├── .gitkeep
    ├── batik_model.h5      # Trained model (not tracked by git)
    └── class_names.json    # Class name list (saved after training)
```

---

## Batik Patterns

| Pattern       | Origin          | Symbolism                                               |
|--------------|-----------------|----------------------------------------------------------|
| **Parang**    | Central Java    | Diagonal waves; strength, sharpness, continuous motion  |
| **Kawung**    | Yogyakarta      | Geometric circles; purity, justice, royal authority     |
| **Mega Mendung** | Cirebon      | Cloud swirls; patience, calm, balance                   |
| **Truntum**   | Solo/Surakarta  | Star-shaped flowers; love regrowing, guidance           |
| **Sidomukti** | Central Java    | Prosperity motif; used in royal wedding ceremonies      |
| **Sekar Jagad** | Java         | Map-like patches; beauty of the world's diversity       |

---

## Dataset

- **Source:** [Kaggle — aldian/batik-kaggle](https://www.kaggle.com/datasets/aldian/batik-kaggle)
- **Format:** JPEG images organized by class folder
- **Split:** 80% training / 20% validation (via `ImageDataGenerator`)

---

## Results

After training and fine-tuning, the model achieves strong classification performance across batik pattern classes. See the notebook for full accuracy/loss curves, confusion matrix, and classification report.

---

## License

MIT License. Dataset credits to the original uploader on Kaggle.
