📋 Project Overview
This project presents a novel CBAM-guided channel pruning framework for efficient osteoporosis classification using knee X-ray images. The methodology achieves 55.9% parameter reduction while maintaining 93.8% diagnostic accuracy, enabling deployment in resource-constrained clinical environments.
🎯 Key Achievements

✅ 55.9% parameter reduction with only 0.4% accuracy loss
✅ 38.9% FLOPs reduction and 35% inference speedup
✅ 93.8% classification accuracy on osteoporosis dataset
✅ Validated across 7 CNN architectures (ResNet-18, VGG16, MobileNetV2, EfficientNet-B0, DenseNet-121, SqueezeNet, ShuffleNetV2)
✅ Outperforms existing pruning methods (L1 norm, iterative magnitude, SE attention)


🔬 Research Highlights
Problem Statement
Deep learning models excel at medical image classification but their computational complexity prevents deployment in resource-limited clinical settings. Existing pruning methods either achieve minimal compression (<5%) or suffer catastrophic accuracy loss (>40%).
Our Solution
CBAM-Guided Intelligent Pruning that:

Uses dual attention mechanisms (channel + spatial) to identify diagnostically critical features
Systematically removes redundant parameters while preserving medical relevance
Implements progressive layer-wise pruning strategy (20-60% across network depth)
Achieves optimal compression-accuracy trade-off

Clinical Impact
Enables deployment of sophisticated osteoporosis detection models on:

📱 Mobile devices
💻 Edge devices
🏥 Resource-constrained clinics
🌍 Rural healthcare facilities


🏗️ Architecture Overview
Input X-ray Image (224×224)
         ↓
    Preprocessing
    (Normalization + Augmentation)
         ↓
    AttentionResNet18
    (CBAM-enhanced)
         ↓
    CBAM Attention Module
    ├─ Channel Attention
    │  (identifies important features)
    └─ Spatial Attention
       (focuses on anatomical regions)
         ↓
    Attention-Guided Pruning
    (removes redundant channels)
         ↓
    Pruned Model
    (55.9% smaller, 35% faster)
         ↓
    Classification Output
    [Healthy | Osteopenia | Osteoporosis]

📊 Dataset
Osteoporosis X-ray Dataset

Total Images: 6,750 knee X-rays
Resolution: 224×224 pixels
Classes: 3 (multiclass classification)

Healthy: 2,890 images
Osteopenia: 1,436 images
Osteoporosis: 2,424 images


Split: 60% train / 20% validation / 20% test

Data Preprocessing

✅ Z-score normalization
✅ Balanced sampling strategy
✅ Data augmentation:

Spatial: Horizontal flip, rotation (±10°), affine transforms
Photometric: Brightness/contrast adjustment (±20%)
Advanced: MixUp augmentation (α=0.2)




🛠️ Technical Details
Model Architecture
Base Model: ResNet-18 with CBAM attention

Original Parameters: 11.7M → Pruned: 4.8M (59% reduction)
Original FLOPs: 3.42B → Pruned: 2.09B (39% reduction)
Original Inference: 12.4ms → Pruned: 8.7ms (30% faster)

CBAM Attention Mechanism
Channel Attention:
M_c(F) = σ(MLP(AvgPool(F)) + MLP(MaxPool(F)))
Spatial Attention:
M_s(F') = σ(Conv_7×7([AvgPool(F'); MaxPool(F')]))
Combined Output:
F' = M_c(F) ⊗ F
F'' = M_s(F') ⊗ F'
Pruning Algorithm
Channel Importance Calculation:
Importance(c_i) = (1/N) * Σ Attention(c_i, x_j)
Progressive Pruning Strategy:

Layer 1: 0-31% pruning (preserve early features)
Layer 2: 30-45% pruning
Layer 3: 45-55% pruning
Layer 4: 50-60% pruning (aggressive on deep layers)


🚀 Installation & Setup
Prerequisites
bashPython >= 3.8
PyTorch >= 1.10.0
CUDA >= 11.0 (for GPU support)
Required Libraries
bashpip install torch torchvision
pip install numpy pandas matplotlib
pip install scikit-learn opencv-python
pip install tensorboard
pip install timm  # For model architectures
Clone Repository
bashgit clone https://github.com/yourusername/osteoporosis-pruning.git
cd osteoporosis-pruning
Install Dependencies
bashpip install -r requirements.txt
```

---

## 📁 Project Structure
```
osteoporosis-pruning/
│
├── data/
│   ├── raw/                    # Original X-ray images
│   ├── processed/              # Preprocessed images
│   └── splits/                 # Train/val/test splits
│
├── models/
│   ├── attention_resnet18.py   # AttentionResNet18 architecture
│   ├── cbam.py                 # CBAM attention module
│   └── pruned_models/          # Saved pruned models
│
├── src/
│   ├── preprocessing.py        # Data preprocessing pipeline
│   ├── training.py             # Model training script
│   ├── pruning.py              # Attention-guided pruning
│   ├── evaluation.py           # Model evaluation metrics
│   └── utils.py                # Utility functions
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_pruning_analysis.ipynb
│   └── 04_results_visualization.ipynb
│
├── configs/
│   ├── train_config.yaml       # Training hyperparameters
│   └── prune_config.yaml       # Pruning configurations
│
├── results/
│   ├── figures/                # Generated visualizations
│   ├── models/                 # Trained model checkpoints
│   └── metrics/                # Performance metrics
│
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── LICENSE                     # License information

💻 Usage
1. Data Preparation
pythonfrom src.preprocessing import prepare_dataset

# Prepare and split dataset
prepare_dataset(
    data_path='data/raw/',
    output_path='data/processed/',
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2
)
2. Train Base Model
pythonfrom src.training import train_model

# Train AttentionResNet18
model = train_model(
    model_name='attention_resnet18',
    data_path='data/processed/',
    epochs=15,
    batch_size=32,
    learning_rate=0.001,
    save_path='results/models/base_model.pth'
)
3. Apply CBAM-Guided Pruning
pythonfrom src.pruning import apply_attention_pruning

# Prune model using CBAM attention
pruned_model = apply_attention_pruning(
    model=model,
    pruning_rate=0.559,  # 55.9% parameter reduction
    importance_threshold=0.3,
    save_path='results/models/pruned_model.pth'
)
4. Evaluate Pruned Model
pythonfrom src.evaluation import evaluate_model

# Evaluate on test set
results = evaluate_model(
    model=pruned_model,
    test_loader=test_loader,
    metrics=['accuracy', 'precision', 'recall', 'f1', 'auc']
)

print(f"Accuracy: {results['accuracy']:.2%}")
print(f"F1-Score: {results['f1']:.2%}")
5. Compare Efficiency
pythonfrom src.evaluation import compare_efficiency

# Compare original vs pruned
comparison = compare_efficiency(
    original_model=model,
    pruned_model=pruned_model,
    input_size=(1, 3, 224, 224)
)

print(f"Parameter Reduction: {comparison['param_reduction']:.1%}")
print(f"FLOP Reduction: {comparison['flop_reduction']:.1%}")
print(f"Inference Speedup: {comparison['speedup']:.1%}")

📈 Results
Model Performance
MetricOriginal ModelPruned ModelChangeAccuracy94.2%93.8%-0.4%Precision94.8%94.3%-0.5%Recall95.1%94.7%-0.4%F1-Score94.9%94.5%-0.4%
Efficiency Gains
MetricOriginalPrunedReductionParameters11.7M4.8M55.9% ↓FLOPs3.42B2.09B38.9% ↓Model Size47.2 MB19.4 MB58.9% ↓Inference Time12.4 ms8.7 ms29.8% ↓GPU Memory892 MB524 MB41.3% ↓
Cross-Architecture Results
ArchitectureParam ReductionAccuracyF1-ScoreSpeedupResNet-1855.9%94.8%95.0%38.1%VGG1650.0%93.8%92.0%39.0%MobileNetV248.2%93.5%93.8%42.3%EfficientNet-B052.7%94.2%94.5%41.8%DenseNet-12149.8%93.9%94.1%40.5%Average50.9%93.7%93.7%41.3%
Comparison with State-of-the-Art
MethodParam ReductionAccuracySpeedupCBAM (Ours)55.9%94.8%38.1%L1 Norm Pruning2.99%83.2%1.2%Iterative Magnitude4.3%54.3%2.4%SE Attention3.93%50.9%2.4%Automatic Attention0.08%89.1%0%
