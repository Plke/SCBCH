# Semantic-Consistent Bidirectional Contrastive Hashing for Noisy Multi-Label Cross-Modal Retrieval

## This project implements the cross-modal hashing method SCBCH.

## Quick Start

1. **Enter the project directory**

   ```bash
   cd ./SCBCH/
   ```

2. **Prepare the dataset**

   - Place dataset into the `./data/` directory.
   - Run the data preprocessing script:
     ```bash
     python ./utils/tools.py
     ```

3. **Generate noisy labels**

   - Run the following command to generate labels with simulated noise:
     ```bash
     python ./noise_label/generate.py
     ```

4. **Train the model**
   - Start training using the specified GPU and parameter:
     ```bash
     python ./train.py --gpus=0 --alpha=0.7 --beta=0.3
     ```

---

## Directory Structure

```
SCBCH/
├── data/                # Raw and processed datasets
├── utils/               # Utility functions
│   └── tools.py         # Data preprocessing script
├── noise_label/         # Noisy label generation
│   └── generate.py      # Script to generate noisy labels
├── train.py             # Model training script
└── README.md            # Project documentation
```
