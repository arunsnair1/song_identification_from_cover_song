# CQTNet Training Pipeline for Indian Music Cover Song Identification - Complete Reference Guide

> **Purpose**: This document is a self-contained reference for the CQTNet training pipeline applied to Indian music cover song identification. It is designed to be pasted into AI assistants (ChatGPT, Claude, etc.) so they can answer any questions about this project with full context.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Constant-Q Transform (CQT) - Deep Explanation](#2-constant-q-transform-cqt---deep-explanation)
3. [Dataset Pipeline - Complete Details](#3-dataset-pipeline---complete-details)
4. [CQTNet Architecture - Layer by Layer](#4-cqtnet-architecture---layer-by-layer)
5. [Training Pipeline - Complete Details](#5-training-pipeline---complete-details)
6. [Evaluation Pipeline - Complete Details](#6-evaluation-pipeline---complete-details)
7. [Key Concepts Glossary](#7-key-concepts-glossary)
8. [Comparison: Original CQTNet vs Our Version](#8-comparison-original-cqtnet-vs-our-version)
9. [Indian Music Specific Context](#9-indian-music-specific-context)
10. [Common Questions and Answers](#10-common-questions-and-answers)
11. [File Structure of the Project](#11-file-structure-of-the-project)
12. [References](#12-references)

---

## 1. Project Overview

### What is Cover Song Identification (CSI)?

Cover Song Identification is a Music Information Retrieval (MIR) task where a system must:
- Take any recording of a song (a "cover" version)
- Identify which original song it is a cover/version of

This is challenging because covers can differ in:
- **Key/pitch**: A cover may be sung in a different key (transposed up or down)
- **Tempo**: The speed may vary significantly
- **Instrumentation**: Different instruments, different arrangements
- **Vocal style**: Different singer, different interpretation, different language even

Yet the underlying melody and harmonic structure remain recognizably similar.

### What is CQTNet?

CQTNet is a Convolutional Neural Network (CNN) designed specifically for cover song identification. It was proposed in the paper "Learning a Representation for Cover Song Identification Using Convolutional Neural Network" (ICASSP 2020).

Key ideas:
1. **Input**: Constant-Q Transform (CQT) spectrograms of audio
2. **Architecture**: Deep CNN (10 convolutional layers) that progressively extracts hierarchical features
3. **Output**: A compact 300-dimensional embedding vector that captures the song's identity
4. **Training strategy**: Trained as a classification task (each original song = one class), but at test time we use the intermediate embedding for similarity comparison

### What is the Goal Here?

We train CQTNet **FROM SCRATCH** on Indian music. This means:
- We do NOT load any pretrained model (no Western-music weights)
- All weights start as random values
- The model learns entirely from Indian music data
- We adapt the final classification layer to match our number of Indian songs

### How Does This Relate to the ISMIR 2025 Paper on Indian Music?

This project implements the baseline system from the paper "Cover Song Identification for Indian Popular Music" (ISMIR 2025). The paper explores:
- Training CQTNet from scratch on Indian music (what we do here)
- Fine-tuning Western-pretrained CQTNet on Indian music
- Adding vocal separation as a preprocessing step
- Using modified triplet loss with data augmentation

### Original CQTNet Training Context

The original CQTNet was trained on SHS100K (Second Hand Songs 100K), a large Western music dataset containing approximately 10,000 unique songs with multiple cover versions each. The Western music in SHS100K is harmony-centric (chord progressions drive the song identity).

### Why Indian Music is Different

Indian popular music differs fundamentally from Western pop/rock:
- **Melody/vocal-dominant**: The vocal melody line is the primary identifier of a song
- **Raga-based influence**: Classical Indian music theory (raga system) influences popular music
- **Less emphasis on chord progressions**: Unlike Western music where chord changes define songs
- **Ornamental singing**: Gamakas, meends, and other vocal ornaments are musically significant
- **Different tonal system**: While using 12 semitones, the microtonal inflections differ

---

## 2. Constant-Q Transform (CQT) - Deep Explanation

### What is a Spectrogram?

Raw audio is a 1D waveform (amplitude vs. time). But music perception is based on **frequencies** (which notes are playing at each moment). A spectrogram converts audio from the time domain to the time-frequency domain, creating a 2D representation showing which frequencies are active at each moment in time.

Think of it as a heat map: x-axis = time, y-axis = frequency, color/brightness = energy at that frequency at that time.

### What is STFT vs Mel Spectrogram vs CQT?

| Feature | STFT | Mel Spectrogram | CQT |
|---------|------|-----------------|-----|
| Frequency spacing | Linear (equal Hz between bins) | Mel-scaled (perceptual) | **Logarithmic (equal musical intervals)** |
| Octave representation | Uneven (higher octaves get more bins) | Somewhat even | **Perfectly even** |
| Musical relevance | Low | Medium | **High** |
| Pitch shift effect | Complex distortion | Non-uniform shift | **Simple vertical translation** |
| Typical use | General audio | Speech recognition | **Music analysis** |

### Why CQT is Ideal for Music

1. **Logarithmic frequency spacing = musical notes evenly spaced**
   - Musical notes follow a logarithmic scale: A4=440Hz, A5=880Hz, A6=1760Hz
   - Each octave doubles in frequency
   - CQT places frequency bins at logarithmically-spaced intervals
   - Result: each semitone gets exactly one bin, every octave is represented equally

2. **84 bins = 7 octaves x 12 bins/octave = 1 bin per semitone**
   - 12 bins per octave = one bin per semitone (the smallest musical interval in standard tuning)
   - 7 octaves covers C1 (32.7 Hz) to B7 (3951 Hz) - encompassing virtually all musical instruments
   - Every note from the lowest bass to the highest treble gets equal representation

3. **Pitch transposition becomes vertical translation (easy for CNNs)**
   - If a cover is sung 2 semitones higher, in CQT this appears as shifting the entire pattern UP by 2 bins
   - This is a simple translation, which CNNs can easily learn to be invariant to
   - In STFT, a pitch shift would cause a non-uniform stretching (much harder for CNNs)

4. **Equal resolution per octave (unlike STFT)**
   - STFT with fixed FFT size: higher frequencies get more bins per octave
   - CQT: exactly 12 bins per octave regardless of whether it is octave 1 or octave 7
   - This means bass notes and treble notes are represented with equal detail

### CQT Computation Details

```python
# The core CQT computation:
cqt_complex = librosa.cqt(y=audio, sr=22050)
# Returns: complex numpy array of shape (84, T)
#   - 84 = frequency bins (7 octaves x 12 bins/octave)
#   - T = number of time frames (depends on audio length and hop_length)

cqt_magnitude = np.abs(cqt_complex)
# np.abs() extracts magnitude, discards phase
# Result shape: (84, T) with all non-negative real values
```

**Key parameters:**
- `sr=22050`: Sample rate in Hz. Standard for music analysis. Nyquist frequency = 11025 Hz.
- `hop_length=512` (default): Number of audio samples between successive CQT frames. At sr=22050, each frame represents 512/22050 = ~23.2ms.
- `n_bins=84` (default): Total frequency bins. 7 octaves x 12 bins/octave.
- `bins_per_octave=12` (default): Frequency resolution. 12 = one bin per semitone.
- Starting frequency: C1 = 32.7 Hz (lowest bin)
- Ending frequency: B7 = 3951 Hz (highest bin)

**Why phase is discarded:**
- Phase contains timing information about where each frequency cycle starts
- Two recordings of the same song will have completely different phase patterns
- Phase is extremely sensitive to recording conditions, microphone placement, etc.
- Magnitude tells us WHAT frequencies are present (the harmonic/melodic content)
- Phase tells us precise cycle alignment (irrelevant for identifying which song it is)
- For cover song identification, we care about "what notes are played" not "exactly when each wave cycle starts"

### Temporal Mean-Pooling

After computing the raw CQT, we apply temporal mean-pooling to reduce dimensionality:

**What it does:** Average every 20 consecutive CQT frames into 1 frame.

**Formula:**
```python
mean_size = 20
new_cqt[:, i] = cqt[:, i*mean_size:(i+1)*mean_size].mean(axis=1)
```

**Why it is done:**
1. **Dimensionality reduction**: A 3-minute song at ~43 frames/sec = ~7700 frames. After pooling: ~385 frames. Much more manageable for the CNN.
2. **Smooths out micro-level timing variations**: Minor tempo fluctuations, vibrato timing, etc. are averaged out.
3. **Memory efficiency**: Smaller arrays = less RAM, faster disk I/O, faster training.
4. **Appropriate time scale**: After pooling, each frame represents ~464ms of audio (20 frames x 23.2ms/frame). This is a musically meaningful time scale (roughly one beat at 130 BPM).

**Example:**
- Input: 3-minute song at 22050 Hz, hop_length=512
- Raw CQT frames: 3*60*22050/512 = ~7744 frames
- After pooling (window=20): 7744/20 = 387 frames
- Compression ratio: 20x fewer time frames
- Time resolution per pooled frame: 20 * 512 / 22050 * 1000 = ~464 ms

**Final output shape:** (84, T_pooled) saved as .npy file, where T_pooled varies by song length.

---

## 3. Dataset Pipeline - Complete Details

### Folder Structure Expected

```
dataset/
  song_001/
    original.mp3
    cover_1.mp3
    cover_2.mp3
  song_002/
    original.mp3
    cover_1.mp3
  song_003/
    original.mp3
    cover_1.mp3
    cover_2.mp3
    cover_3.mp3
  ...
```

**Rules:**
- Each folder = one class (one original song)
- All files in a folder = different versions/covers of that song
- Folder names can be anything (they are sorted alphabetically and assigned integer class IDs: 0, 1, 2, ...)
- You need at least 2 files per folder (otherwise there is no cover relationship to learn)
- More covers per song = better training signal
- File format: MP3, WAV, FLAC, or any format librosa can read

### Naming Convention for .npy Files

After CQT extraction, files are saved as: `{class_id}_{version_id}.npy`

Examples:
- `0_0.npy` = Class 0, version 0 (first file in first song folder)
- `0_1.npy` = Class 0, version 1 (second file in first song folder)
- `1_0.npy` = Class 1, version 0 (first file in second song folder)
- `244_3.npy` = Class 244, version 3 (fourth file in the 245th song folder)

### CQT Feature Extraction Function

```python
def compute_cqt_features(mp3_path, sr=22050, mean_size=20):
    # Step 1: Load audio, resample to 22050 Hz, convert stereo to mono
    data, sr = librosa.load(mp3_path, sr=sr)
    # data shape: (num_samples,) - 1D array of audio samples
    
    # Step 2: Compute CQT magnitude (discard phase)
    cqt = np.abs(librosa.cqt(y=data, sr=sr))
    # cqt shape: (84, num_frames)
    
    # Step 3: Temporal mean-pooling
    height, length = cqt.shape  # height=84, length=num_frames
    new_length = length // mean_size
    new_cqt = np.zeros((height, new_length), dtype=np.float64)
    for i in range(new_length):
        new_cqt[:, i] = cqt[:, i*mean_size:(i+1)*mean_size].mean(axis=1)
    # new_cqt shape: (84, new_length)
    
    return new_cqt
```

### PyTorch Dataset Class

The `CQTDataset` class handles loading and preprocessing individual samples:

```python
class CQTDataset(Dataset):
    def __init__(self, file_list, out_length=200, is_training=True):
        self.file_list = file_list      # List of (filepath, label) tuples
        self.out_length = out_length    # Target temporal dimension
        self.is_training = is_training  # Controls random vs center crop

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath, label = self.file_list[idx]
        # ... preprocessing pipeline ...
        return data_tensor, label
```

**Key methods:**
- `__len__()`: Returns total number of audio files in the dataset
- `__getitem__(idx)`: Loads one .npy file, preprocesses it, returns (tensor, label)

### Preprocessing Pipeline in __getitem__

Each sample goes through these steps in order:

**Step 1: Load .npy file**
```python
cqt = np.load(filepath)  # Shape: (84, T) where T varies per file
```

**Step 2: Transpose to (T, 84)**
```python
cqt = cqt.T  # Now shape: (T, 84) - for easier time-axis cropping
```
Why: Makes random cropping along the time axis (axis 0) straightforward.

**Step 3: Normalize: (x - mean) / std**
```python
mean = cqt.mean()
std = cqt.std()
if std > 0:
    cqt = (cqt - mean) / std
```
Why: Centers data around 0 with standard deviation of 1. Neural networks train much better when inputs are normalized. Without this, songs recorded at different volumes would confuse the network.

**Step 4: Random crop (train) or center crop (test) to fixed out_length**
```python
T = cqt.shape[0]
if T >= out_length:
    if is_training:
        start = random.randint(0, T - out_length)  # Random position
    else:
        start = (T - out_length) // 2  # Center position
    cqt = cqt[start:start + out_length, :]
```
Why: The CNN requires fixed-size input. Random cropping during training acts as data augmentation (the model sees different segments each epoch). Center cropping during evaluation ensures deterministic results.

**Step 5: Pad with zeros if shorter than out_length**
```python
if T < out_length:
    padding = np.zeros((out_length - T, 84))
    cqt = np.concatenate([cqt, padding], axis=0)
```
Why: Short songs need padding to reach the required length. Zero-padding is used because after normalization, zero represents "no signal."

**Step 6: Transpose back to (84, out_length)**
```python
cqt = cqt.T  # Shape: (84, out_length)
```

**Step 7: Add channel dimension: (1, 84, out_length)**
```python
cqt = cqt[np.newaxis, :, :]  # Shape: (1, 84, out_length)
```
Why: PyTorch Conv2d expects input shape (batch, channels, height, width). The CQT spectrogram is like a grayscale image with 1 channel.

**Step 8: Convert to torch.FloatTensor**
```python
data_tensor = torch.FloatTensor(cqt)
```

### DataLoader

The DataLoader wraps the Dataset and handles:
- **Batching**: Groups multiple samples into a batch (e.g., batch_size=32)
- **Shuffling**: Randomizes order each epoch (important for training, disabled for testing)
- **Parallel loading**: Can use multiple CPU workers for faster data loading

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,       # 32 samples per gradient update
    shuffle=True,        # Randomize order each epoch
    num_workers=2,       # Parallel data loading
    drop_last=True       # Drop incomplete final batch (helps BatchNorm stability)
)
```

### Multi-size Training (Advanced)

The original CQTNet paper alternates between out_length=200, 300, and 400 across training epochs. This makes the model robust to different temporal scales and ensures the AdaptiveMaxPool layer works well regardless of input length.

Implementation idea:
```python
out_lengths = [200, 300, 400]
for epoch in range(num_epochs):
    current_out_length = out_lengths[epoch % 3]
    # Recreate dataset with new out_length
```

---

## 4. CQTNet Architecture - Layer by Layer

### Overview

```
Input: (batch, 1, 84, time)
  -> Block 1: conv0 + conv1 + pool1
  -> Block 2: conv2 + conv3 + pool3
  -> Block 3: conv4 + conv5 + pool5
  -> Block 4: conv6 + conv7 + pool7
  -> Block 5: conv8 + conv9 (no pool)
  -> AdaptiveMaxPool2d((1,1)) -> (batch, 512, 1, 1)
  -> Flatten -> (batch, 512)
  -> fc0: Linear(512, 300) -> THE EMBEDDING (used for retrieval)
  -> fc1: Linear(300, num_classes) -> CLASSIFICATION HEAD (training only)
```

### Block 1: Initial Feature Extraction

**conv0: Conv2d(1, 32, kernel=(12,3), dilation=(1,1), padding=(6,0), bias=False)**
- Input channels: 1 (single-channel CQT spectrogram)
- Output channels: 32 (32 different frequency-time pattern detectors)
- Kernel size (12, 3):
  - 12 in frequency = spans exactly 1 octave in CQT (12 bins = 12 semitones = 1 octave)
  - This captures octave relationships and harmonics (a note and its harmonic an octave above)
  - 3 in time = looks at 3 consecutive time frames (~1.4 seconds of pooled audio)
- padding=(6, 0): Pads 6 on each side of the frequency axis to maintain frequency dimension size
- bias=False: No bias because the following BatchNorm has its own learnable bias (beta parameter). Having both would be redundant.

**norm0: BatchNorm2d(32)**
- Normalizes all 32 feature maps to mean=0, std=1 across the batch
- Then applies learnable parameters: gamma * normalized + beta
- Why: Prevents internal covariate shift, enables higher learning rates, provides mild regularization

**relu0: ReLU (Rectified Linear Unit)**
- f(x) = max(0, x)
- Introduces non-linearity. Without it, stacking linear layers would still be a single linear transformation.
- Creates sparse activations (many zeros), which helps with computation and learning.

**conv1: Conv2d(32, 64, kernel=(13,3), dilation=(1,2), bias=False)**
- Increases channels from 32 to 64
- Kernel (13, 3) with dilation=(1, 2):
  - 13 in frequency: slightly more than 1 octave
  - 3 in time with dilation=2: the kernel samples positions [t, t+2, t+4] instead of [t, t+1, t+2]
  - Effective receptive field in time = 5 frames (sees a wider temporal context)
  - Why dilation: Increases temporal context without increasing parameter count. Gets the receptive field of a 5-wide kernel with only 3 parameters in that dimension.

**norm1: BatchNorm2d(64)**, **relu1: ReLU**

**pool1: MaxPool2d((1,2), stride=(1,2), padding=(0,1))**
- Kernel (1, 2) with stride (1, 2): Only pools in the TIME dimension, not frequency
- Why only time: We want to preserve full frequency resolution (all 84 bins convey musical pitch info)
- Effect: Reduces temporal dimension by approximately half
- Adds translation invariance in the time direction (minor timing shifts do not change the output)
- padding=(0,1): Handles cases where time dimension is odd

### Block 2

**conv2: Conv2d(64, 64, kernel=(13,3), dilation=(1,1), bias=False)**
- Same channel count (64). Large frequency kernel (13) continues to capture cross-octave patterns.

**norm2, relu2**

**conv3: Conv2d(64, 64, kernel=(3,3), dilation=(1,2), bias=False)**
- Smaller frequency kernel now (3 instead of 13)
- Why smaller: After the initial layers captured broad octave-spanning patterns, now we look for fine-grained frequency patterns (intervals of 2-3 semitones)
- dilation=(1,2): Again expands temporal receptive field without more parameters

**norm3, relu3**

**pool3: MaxPool2d((1,2), stride=(1,2), padding=(0,1))**
- Another time-only pooling, further reducing temporal dimension by ~half

### Block 3: Channel Increase to 128

**conv4: Conv2d(64, 128, kernel=(3,3), dilation=(1,1), bias=False)**
- Doubles channels from 64 to 128
- More channels = more different patterns can be detected simultaneously
- At this depth, the network detects more abstract/complex patterns

**norm4, relu4**

**conv5: Conv2d(128, 128, kernel=(3,3), dilation=(1,2), bias=False)**
- Maintains 128 channels, uses dilation for temporal context

**norm5, relu5**

**pool5: MaxPool2d((1,2), stride=(1,2), padding=(0,1))**

### Block 4: Channel Increase to 256

**conv6: Conv2d(128, 256, kernel=(3,3), dilation=(1,1), bias=False)**
- Doubles channels to 256

**norm6, relu6**

**conv7: Conv2d(256, 256, kernel=(3,3), dilation=(1,2), bias=False)**

**norm7, relu7**

**pool7: MaxPool2d((1,2), stride=(1,2), padding=(0,1))**

### Block 5: Channel Increase to 512 (Final Convolutional Block)

**conv8: Conv2d(256, 512, kernel=(3,3), dilation=(1,1), bias=False)**
- Doubles channels to 512 (final channel count)
- At this depth, features are highly abstract representations of musical structure

**norm8, relu8**

**conv9: Conv2d(512, 512, kernel=(3,3), dilation=(1,2), bias=False)**
- Final convolutional layer, maintains 512 channels

**norm9, relu9**

### Global Pooling

**AdaptiveMaxPool2d((1,1))**
- Forces output to exactly (batch, 512, 1, 1) regardless of input spatial size
- WHY this is critical: It enables variable-length input!
  - Whether input is 200 frames, 300 frames, or 400 frames, the output is always (batch, 512, 1, 1)
  - It automatically computes the required kernel size and stride to produce exactly 1x1 output
  - This is what makes multi-size training possible
- HOW it works: For an input of size (H, W), it uses a kernel of size (H, W) and stride (H, W), effectively taking the global maximum across all spatial positions

### Fully Connected Layers

**fc0: Linear(512, 300) - THE EMBEDDING LAYER**
- Input: 512-dimensional flattened feature vector
- Output: 300-dimensional embedding vector
- This 300-dim vector IS the song's identity representation
- Used for retrieval at test time (this is what gets compared between songs)
- Why 300: Trade-off between expressiveness and compactness. The original paper chose 300 through experimentation.

**fc1: Linear(300, num_classes) - CLASSIFICATION HEAD**
- Input: 300-dimensional embedding
- Output: num_classes logits (one score per song class)
- Only used during training to provide the learning signal via cross-entropy loss
- At test time, we extract the output of fc0 and completely ignore fc1
- num_classes = number of unique songs:
  - 5 for demo mode
  - 245 for the CoversIndian dataset
  - 10000 for the original SHS100K dataset

### Design Patterns in the Architecture

- **Channel progression**: 1 -> 32 -> 64 -> 64 -> 128 -> 256 -> 512
  - Gradually increasing abstraction capacity
  - Lower layers: few channels, detect simple patterns (edges, harmonics)
  - Deeper layers: many channels, detect complex patterns (melodic phrases, chord sequences)

- **Alternating dilation pattern**: Each block has one conv with dilation=(1,1) and one with dilation=(1,2)
  - First conv: standard local patterns
  - Second conv: expanded temporal context (sees further in time without more parameters)

- **Only time-axis pooling**: Frequency is never downsampled by pooling
  - Frequency dimension is gradually reduced only by convolution (kernel extends beyond edges)
  - This preserves pitch information at all stages

- **Total parameters**: Approximately 4.3 million (relatively small for a CNN)
  - Model size in memory: ~17 MB (float32)
  - Small enough to train on a single consumer GPU

---

## 5. Training Pipeline - Complete Details

### Loss Function: Cross-Entropy

**What it is:** Cross-entropy measures how different the model's predicted probability distribution is from the true distribution (one-hot label).

**Formula:** `Loss = -log(P(correct_class))`

**How it works:**
- Input: logits of shape (batch, num_classes) + integer labels of shape (batch,)
- PyTorch's CrossEntropyLoss internally applies softmax to convert logits to probabilities, then computes negative log-likelihood
- If model predicts 90% for correct class: loss = -log(0.9) = 0.105 (low, good)
- If model predicts 10% for correct class: loss = -log(0.1) = 2.303 (high, bad)
- If model predicts 1% for correct class: loss = -log(0.01) = 4.605 (very high, very bad)

**Why it works for embeddings:**
- To correctly classify "this CQT belongs to song A", the fc0 layer must produce embeddings where:
  - All covers of song A map to similar 300-dim vectors (so fc1 can classify them the same)
  - Songs B, C, D map to DIFFERENT 300-dim vectors (so fc1 can distinguish them)
- This naturally creates an embedding space where covers cluster together and different songs are separated

### Optimizer: Adam (Adaptive Moment Estimation)

**What it does:** Updates model parameters to minimize the loss function.

**How it works:**
- Maintains two running averages for each parameter:
  - First moment (mean of gradients) - provides momentum, smooths noisy gradients
  - Second moment (mean of squared gradients) - provides per-parameter adaptive learning rate
- Parameters with consistently large gradients get smaller effective learning rates (already moving fast)
- Parameters with small/noisy gradients get larger effective learning rates (need more push)

**Default settings:**
- `lr=1e-3` (0.001): Base learning rate. Good default for most deep learning tasks.
- `weight_decay=1e-4` (0.0001): L2 regularization. Adds penalty `weight_decay * sum(param^2)` to loss. Prevents weights from growing too large, which reduces overfitting.

### Learning Rate Scheduler: ReduceLROnPlateau

**What it does:** Monitors training loss and reduces learning rate when progress stalls.

**How it works:**
- Monitors a metric (training loss, in our case)
- If the loss does not improve for `patience` epochs, multiply the learning rate by `factor`
- This helps escape flat regions or saddle points in the loss landscape

**Default settings:**
- `patience=5`: Wait 5 epochs without improvement before reducing
- `factor=0.5`: Halve the learning rate each time
- `mode='min'`: "Improvement" means the loss decreased

**Example trajectory:**
- Epochs 1-20: lr = 0.001 (loss decreasing)
- Epochs 21-26: loss plateaus, no improvement for 5 epochs
- Epoch 27: scheduler reduces lr to 0.0005
- Epochs 27-40: loss starts decreasing again with smaller steps

### Training Loop Steps (Per Epoch)

```python
model.train()  # Enable training-mode BatchNorm (uses batch statistics)

for data, labels in train_loader:
    # 1. Move data to device (GPU/CPU)
    data = data.to(device)      # Shape: (batch, 1, 84, out_length)
    labels = labels.to(device)  # Shape: (batch,)
    
    # 2. Zero gradients from previous iteration
    optimizer.zero_grad()
    # WHY: PyTorch accumulates gradients by default. Without zeroing,
    # gradients from batch N would add to gradients from batch N+1.
    
    # 3. Forward pass
    logits, embedding = model(data)
    # logits: (batch, num_classes) - raw class scores
    # embedding: (batch, 300) - song representation
    
    # 4. Compute loss
    loss = criterion(logits, labels)
    # Single scalar value measuring prediction error
    
    # 5. Backward pass (backpropagation)
    loss.backward()
    # Computes gradient of loss w.r.t. every learnable parameter
    # Uses chain rule of calculus applied layer-by-layer from output to input
    # After this: every parameter.grad contains its gradient
    
    # 6. Optimizer step
    optimizer.step()
    # Updates each parameter: param = param - lr * adjusted_gradient
    # Adam uses adaptive per-parameter learning rates (not just raw gradient)

# After all batches in epoch:
scheduler.step(avg_loss)  # Check if LR should be reduced
```

### Why Classification Trains Good Embeddings (The Core Trick)

This is the fundamental insight of CQTNet's approach:

1. We train the model as a **classifier**: "Given this CQT, which of N songs is it?"
2. The classification layer (fc1) provides the learning signal via cross-entropy
3. BUT the actual useful output is the **intermediate embedding** from fc0

**Why this works:** For fc1 to correctly separate N classes, fc0 must produce an embedding space where:
- All covers of song A cluster tightly together (so fc1 can draw a decision boundary around them)
- Clusters for different songs are well-separated (so fc1 can distinguish between them)

This is exactly what we need for retrieval: similar songs should have similar embeddings.

**At test time:** We throw away fc1 entirely and just use the 300-dim fc0 output. This works for songs NEVER seen during training because the embedding space has learned general musical similarity, not just memorized the training songs.

### Training Hyperparameters Summary

| Parameter | Demo Value | Real Training Value | Purpose |
|-----------|-----------|-------------------|---------|
| NUM_EPOCHS | 10 | 50-200 | Total training iterations over dataset |
| LEARNING_RATE | 1e-3 | 1e-3 | Step size for gradient updates |
| WEIGHT_DECAY | 1e-4 | 1e-4 | L2 regularization (prevents overfitting) |
| BATCH_SIZE | 4 | 32-64 | Samples per gradient update |
| OUT_LENGTH | 200 | 200/300/400 | Fixed temporal input length |
| LR_PATIENCE | 5 | 5-10 | Epochs before LR reduction |
| LR_FACTOR | 0.5 | 0.5 | LR multiplication factor |

---

## 6. Evaluation Pipeline - Complete Details

### Overview

Evaluation uses the trained model completely differently from training:
- Training: CQT -> model -> classification logits -> cross-entropy loss -> backprop
- Evaluation: CQT -> model -> 300-dim embedding -> normalize -> compare -> rank -> MAP

### Step 1: Extract Embeddings

```python
model.eval()  # Switch BatchNorm to use running statistics (not batch stats)
              # Also disables dropout if present

all_embeddings = []
with torch.no_grad():  # Disable gradient computation (saves ~50% memory, faster)
    for data, labels in data_loader:
        data = data.to(device)
        logits, embedding = model(data)
        # We IGNORE logits! Only take the 300-dim embedding from fc0.
        all_embeddings.append(embedding.cpu().numpy())

embeddings = np.concatenate(all_embeddings, axis=0)  # Shape: (N, 300)
```

**Key differences from training mode:**
- `model.eval()`: BatchNorm uses running mean/variance (accumulated during training) instead of batch statistics. This ensures consistent behavior regardless of batch size or composition.
- `torch.no_grad()`: No gradient computation needed (not doing backprop). Saves memory and speeds up computation by ~50%.
- We only use the embedding output, not the logits.

### Step 2: L2 Normalization

```python
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)  # Shape: (N, 1)
embeddings_normalized = embeddings / norms  # Shape: (N, 300)
# After this: every embedding vector has unit length (lies on unit hypersphere)
```

**Why normalize:**
- After L2 normalization, all vectors have length 1 (they lie on the surface of a 300-dimensional unit sphere)
- Cosine similarity between two unit vectors = their dot product (no division needed)
- Removes the effect of magnitude: only the DIRECTION of the embedding matters
- Two songs with embeddings pointing in the same direction are similar, regardless of how "strongly activated" each was

**Mathematical connection:**
- Cosine similarity formula: `cos(a, b) = (a . b) / (||a|| * ||b||)`
- When ||a|| = ||b|| = 1: `cos(a, b) = a . b` (just a dot product!)
- This makes computing the similarity matrix extremely fast

### Step 3: Similarity Matrix

```python
similarity_matrix = embeddings_normalized @ embeddings_normalized.T
# Matrix multiplication: (N, 300) @ (300, N) = (N, N)
```

**What this produces:**
- An N x N matrix where entry [i, j] = cosine similarity between song i and song j
- Diagonal entries = 1.0 (a song is perfectly similar to itself)
- Values range from -1.0 (opposite direction) to 1.0 (same direction)
- Same-class pairs (covers) should be close to 1.0
- Different-class pairs should be close to 0.0 or negative

**Visual interpretation (heatmap):**
- If the model has learned well, you see a **block-diagonal** pattern
- Bright squares on the diagonal = covers of the same song cluster together
- Dark off-diagonal regions = different songs are well separated

### Step 4: Mean Average Precision (MAP)

MAP is the standard evaluation metric for retrieval tasks, including cover song identification.

**Computation process:**

For each query song i:
1. Rank all other songs by similarity to i (highest similarity first)
2. Go through the ranked list. Each time you encounter a true cover (same class):
   - Compute precision@k = (number of covers found so far) / (current rank position)
3. Average Precision (AP) for this query = mean of all precision@k values
4. MAP = mean of AP across all queries

**Example:**
- Query: Song A. True covers of A are at database positions [3, 7, 15].
- After ranking by similarity, suppose covers are found at ranks 1, 4, and 6:
  - Cover at rank 1: precision@1 = 1/1 = 1.000
  - Cover at rank 4: precision@4 = 2/4 = 0.500
  - Cover at rank 6: precision@6 = 3/6 = 0.500
  - AP = (1.000 + 0.500 + 0.500) / 3 = 0.667

**Interpretation:**
- MAP = 1.0: Perfect. All covers always ranked at the very top.
- MAP ~ 1/num_classes: Random baseline. No better than chance.
- MAP > 0.8: Excellent for small datasets.
- MAP 0.5-0.8: Good, model is working but room for improvement.
- MAP < 0.3: Poor, model is barely learning.

### Additional Metrics

**Rank1 (Mean Rank of First Correct Cover):**
- For each query, what rank is the first true cover at?
- Lower is better (ideal = 1, meaning the first result is always correct)
- More intuitive than MAP for non-experts

**Top10 (Precision@10):**
- Of the top 10 results returned for each query, what fraction are true covers?
- Practical metric: "If I look at the first page of results, how many are correct?"

### Complete Evaluation Code

```python
def compute_map(similarity_matrix, labels):
    N = len(labels)
    per_query_ap = []
    
    for i in range(N):
        query_label = labels[i]
        similarities = similarity_matrix[i].copy()
        similarities[i] = -1.0  # Exclude self-match
        
        relevant_mask = (labels == query_label)
        relevant_mask[i] = False
        num_relevant = relevant_mask.sum()
        
        if num_relevant == 0:
            continue
        
        ranked_indices = np.argsort(-similarities)  # Descending order
        
        hits = 0
        sum_precision = 0.0
        for rank, idx in enumerate(ranked_indices):
            if idx == i:
                continue
            if relevant_mask[idx]:
                hits += 1
                precision_at_k = hits / (rank + 1)
                sum_precision += precision_at_k
            if hits == num_relevant:
                break
        
        ap = sum_precision / num_relevant
        per_query_ap.append(ap)
    
    return np.mean(per_query_ap), per_query_ap
```

---

## 7. Key Concepts Glossary

**Embedding**: A fixed-size vector representation (here: 300 numbers) that captures the essence of the input. Songs with similar melodies/harmonics should have similar embeddings. The embedding is a point in 300-dimensional space.

**Epoch**: One complete pass through the entire training dataset. If you have 490 training samples and batch_size=32, one epoch = ceiling(490/32) = 16 batches = 16 gradient updates.

**Batch**: A subset of training samples processed together for one gradient update. Larger batches give more stable gradient estimates but require more memory.

**Batch Size**: The number of samples in each batch. Trade-off: larger = more stable gradients but more memory. Typical values: 16, 32, 64.

**Gradient**: The direction and magnitude of change needed for each parameter to reduce the loss. Computed via backpropagation. Points "uphill" in loss landscape, so we move in the opposite direction.

**Backpropagation**: Algorithm that efficiently computes gradients for all parameters using the chain rule of calculus. Propagates error signal backward from the loss, through each layer, to the inputs.

**Learning Rate**: How big a step to take in the gradient direction when updating weights. Too high = unstable (loss explodes). Too low = painfully slow convergence. Typical starting value: 0.001 for Adam.

**Weight Decay (L2 Regularization)**: Adds a penalty proportional to the square of the weights: `loss += weight_decay * sum(w^2)`. Prevents weights from growing too large. Acts as regularization against overfitting.

**Overfitting**: When the model memorizes training data but fails on new/unseen data. Signs: training loss near 0, but test/validation loss is high or increasing. The model has learned noise specific to training samples rather than general patterns.

**Regularization**: Techniques to prevent overfitting. Examples include weight decay (L2 penalty), dropout (randomly zero out neurons), data augmentation (create variations of training data), and early stopping (stop training before overfitting occurs).

**Receptive Field**: How much of the original input each neuron in a deeper layer can "see." Influenced by kernel sizes, dilation, and pooling. Deeper layers have larger receptive fields and thus capture more global patterns.

**Feature Map**: The output of a single convolutional filter applied to the input. A 2D grid of activations showing where that particular pattern was detected. A layer with 64 output channels produces 64 feature maps.

**Cosine Similarity**: Measures the angle between two vectors. Range: [-1, 1]. Value of 1 = identical direction (maximally similar). Value of 0 = perpendicular (unrelated). Value of -1 = opposite direction (maximally dissimilar).

**Dilation**: Inserts gaps between kernel elements in a convolution. dilation=2 means the kernel skips every other position. Increases receptive field without increasing parameter count. Effective kernel size = actual_size + (actual_size - 1) * (dilation - 1).

**Transfer Learning**: Using weights trained on one task/dataset for a different but related task. NOT what we do here (we train from scratch).

**Fine-tuning**: A form of transfer learning where you load pretrained weights and continue training (usually with a smaller learning rate) on new data. Different from training from scratch because the model starts with knowledge from the original task.

**Softmax**: Converts a vector of raw scores (logits) into a probability distribution. Each output is between 0 and 1, and all outputs sum to 1. Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)).

**Logits**: Raw, unnormalized scores output by the network before softmax is applied. Can be any real number (positive, negative, or zero).

**State Dict**: A Python dictionary mapping each layer name to its parameter tensor. Used to save and load model weights. Does not save architecture, only parameter values.

**AdaptiveMaxPool**: A pooling operation where you specify the desired output size rather than the kernel size. The kernel and stride are automatically computed to achieve the target output dimensions.

**Inference**: Using a trained model to make predictions on new data (as opposed to training it). No gradient computation is needed during inference.

**Hyperparameters**: Settings that control the training process but are NOT learned by the model. Examples: learning rate, batch size, number of epochs, weight decay. Must be chosen by the practitioner.

---

## 8. Comparison: Original CQTNet vs Our Version

| Aspect | Original CQTNet (ICASSP 2020) | Our Version (Indian Music) |
|--------|-------------------------------|---------------------------|
| Dataset | SHS100K (Western, ~10,000 songs) | Indian music (e.g., 245 songs from CoversIndian) |
| Music type | Harmony-centric Western pop/rock | Melody/vocal-dominant Indian popular music |
| num_classes in fc1 | 10,000 | Variable (5 for demo, 245 for real) |
| Weight initialization | Random (trained from scratch on SHS100K) | Random (training from scratch on Indian data) |
| Loss function | Cross-Entropy | Cross-Entropy (identical) |
| Optimizer | Adam | Adam (identical) |
| Multi-size training | out_length cycles through 200, 300, 400 | out_length=200 (simplified, expandable) |
| SpecAugment | Yes (in later versions) | Not included (can be added as augmentation) |
| Batch size | 32 | 4 (demo), 32 (real training) |
| Architecture (CNN layers) | 10 conv layers + 2 FC | Identical (same layers, same dimensions) |
| Embedding dimension | 300 | 300 (identical) |
| Temporal mean-pooling | Window = 20 | Window = 20 (identical) |
| CQT parameters | 84 bins, hop=512, sr=22050 | 84 bins, hop=512, sr=22050 (identical) |
| Training data size | ~100,000 recordings | ~490 recordings (much smaller) |
| Training duration | Many hours on GPU | 1-3 hours on single GPU |
| Reported MAP on own test set | ~0.85 on SHS100K test | Expected 0.3-0.7 from scratch on Indian data |

### Key Differences Explained

**Why fewer classes matters:**
- With only 245 classes instead of 10,000, the classification task is "easier" but we have far fewer training examples
- Risk of overfitting is much higher with small datasets
- Weight decay becomes more important for regularization

**Why training from scratch instead of fine-tuning:**
- Proves the model can learn Indian music features independently
- Eliminates any bias from Western music features
- Serves as a true baseline for comparison
- Shows what is achievable without any external pretrained knowledge

**What stays the same:**
- The entire CNN architecture (all layer dimensions, kernel sizes, dilations)
- The training strategy (classification then use embeddings for retrieval)
- The evaluation protocol (L2 normalize, cosine similarity, MAP)
- The CQT preprocessing pipeline

---

## 9. Indian Music Specific Context

### Characteristics of Indian Popular Music

- **Melody-dominant**: The vocal melody line is the primary identifier of a song. Listeners recognize songs by their tune more than their chord progression.
- **Vocal-dominant**: The singing voice carries most of the musical identity. Instrumental arrangements are secondary.
- **Raga-based influence**: Indian classical music theory (raga system with specific ascending/descending note patterns) strongly influences popular music. Many film songs are based on specific ragas.
- **Ornamental singing**: Gamakas (oscillations), meends (glides between notes), and other micro-pitch ornaments are musically significant and carry identity information.
- **Less emphasis on chord progressions**: Unlike Western pop where the chord sequence (e.g., I-V-vi-IV) defines a song, Indian songs are defined more by melodic contour.
- **Heterophonic texture**: Multiple instruments often play variations of the same melody simultaneously, rather than independent harmonic parts.

### How This Differs from Western Music (for CSI purposes)

- **Western covers** might completely change the melody but keep the same chord progression. The original CQTNet learned to match harmonic (chord) patterns.
- **Indian covers** typically preserve the original vocal melody closely (the melody IS the song). Changes are more in ornamentation, tempo, and instrumentation.
- This suggests that a model trained on Indian music should focus more on melodic line detection than harmonic structure detection.

### Vocal Separation for Indian Music CSI

The ISMIR 2025 paper shows that vocal separation significantly helps Indian CSI:
- Separating the vocal track from the mix removes instrumental "noise" that confuses the model
- Since Indian song identity is primarily in the vocals, isolating them gives cleaner input
- This is less impactful for Western music where harmony (in accompaniment) is equally important

### CoversIndian Dataset

- **Size**: 245 song pairs (each pair has at least 2 versions)
- **Languages**: Hindi and Malayalam (two major Indian languages)
- **Source**: Indian film music (Bollywood and Malayalam cinema)
- **Characteristics**: Professional studio recordings with varying production quality across decades

### Results from the ISMIR 2025 Paper

These results provide context for what performance to expect:

| Method | MAP |
|--------|-----|
| CQTNet pretrained on Western (SHS100K), tested on Indian | 0.7238 |
| CQTNet fine-tuned on Indian data | 0.7379 |
| CQTNet + Vocal Separation | 0.9001 |
| CQTNet + Modified Triplet Loss + Data Augmentation + Vocal Separation | 0.9234 (best) |

**What these numbers mean for our project:**
- Our from-scratch training (no pretrained weights) will likely achieve lower MAP than the pretrained baseline (0.7238)
- This is expected because we have much less training data and no prior knowledge
- The gap shows the value of the larger Western pretraining dataset
- Adding vocal separation could dramatically improve our results too
- The best result (0.9234) shows what is achievable with all enhancements combined

### Why CQT Still Works for Indian Music

Even though Indian music is melody-dominant rather than harmony-dominant:
- CQT captures melodic patterns (sequences of single notes over time)
- The vertical axis still represents pitch, so melodic contours appear as patterns in the CQT
- Temporal patterns in CQT encode rhythmic and melodic sequences
- The CNN can learn to focus on whatever patterns (harmonic OR melodic) are most discriminative for the training data

---

## 10. Common Questions and Answers

### About the Approach

**Q: Why not just use the pretrained Western model directly?**

A: The pretrained model learned features optimized for Western music (harmony-centric, chord-progression-based). Indian music is melody/vocal-dominant. Training from scratch allows the model to learn features specifically suited to Indian music characteristics. Additionally, training from scratch provides a clean baseline to measure the value of domain-specific training versus transfer from Western music.

**Q: Why is the dataset small (245 songs) compared to SHS100K (100,000)?**

A: Curated Indian music cover datasets are rare. Creating a cover song dataset requires carefully identifying which songs are covers of each other, which requires musical expertise. CoversIndian is one of the first systematic datasets for Indian cover song identification. Small datasets require careful regularization (weight decay, data augmentation) to avoid overfitting.

**Q: Can I use this for any language/genre of Indian music?**

A: Yes. The CQT captures frequency patterns regardless of language. Hindi, Malayalam, Tamil, Telugu, Kannada, Bengali - all work. The model learns from harmonic/melodic structure (which frequencies are active over time), not from lyrics or language-specific features. However, performance may vary across genres if the training data is dominated by one genre.

**Q: What is the difference between training from scratch and fine-tuning?**

A: From scratch: all weights start as random values. The model learns everything exclusively from Indian data. Fine-tuning: start from pretrained Western weights (already encoding musical knowledge from SHS100K), then continue training with Indian data to adapt those features. Fine-tuning usually converges faster and may perform better with limited data, but training from scratch proves the model can learn Indian-specific features independently.

### About Hardware and Practical Concerns

**Q: What hardware do I need?**

A: GPU strongly recommended for real training. Google Colab free tier (T4 GPU) works well. For demo mode with synthetic data, CPU is fine (finishes in seconds). Real training on 245 songs with a T4 GPU takes roughly 1-3 hours for 100 epochs.

**Q: How long does real training take?**

A: Approximate training times for 100 epochs on the CoversIndian dataset (245 songs, ~490 files):
- Google Colab T4 GPU: 1-3 hours
- Consumer laptop CPU: 10-20+ hours (not recommended)
- Desktop RTX 3070+: Under 1 hour
- Training time scales roughly linearly with dataset size and number of epochs.

**Q: What if my MP3 files have different sample rates?**

A: librosa.load() automatically resamples everything to the target sample rate (22050 Hz by default). No manual conversion needed. It also automatically converts stereo to mono. Any audio format that libsndfile or ffmpeg can read is supported.

### About the Model

**Q: Why 300 dimensions for the embedding?**

A: This is a trade-off between expressiveness and compactness chosen by the original paper authors through experimentation. Larger (e.g., 512) might capture more nuance but risks overfitting on small datasets and increases computation for similarity search. Smaller (e.g., 128) might lose information needed to distinguish between similar songs. 300 was empirically found to work well for cover song identification.

**Q: How do I know if the model is overfitting?**

A: Monitor training loss vs. a held-out validation loss:
- Both decreasing together = model is learning well (good generalization)
- Training loss decreasing but validation loss increasing = overfitting
- Training loss plateaus at a high value = underfitting (model too simple or learning rate too low)

Solutions for overfitting:
- Increase weight_decay (stronger L2 regularization)
- Add data augmentation (pitch shift, time stretch, noise injection)
- Use dropout layers
- Train for fewer epochs (early stopping)
- Get more training data

**Q: Why does the model output BOTH logits AND embeddings?**

A: During training, we need logits for the cross-entropy loss (the classification training signal). During testing/inference, we only need the 300-dim embedding (for retrieval via similarity comparison). The model returns both so that the training loop can compute loss while also giving access to the embedding when needed.

**Q: What does the 300-dim embedding actually represent?**

A: It is a learned compact representation of the song's musical identity - its melodic patterns, harmonic structure, rhythmic characteristics, and timbral qualities, all encoded into 300 numbers. The training process forces the network to encode whatever information is most useful for distinguishing between different songs while grouping covers of the same song together. It is not interpretable in a human-readable way (you cannot say "dimension 42 represents the key signature"), but the geometric relationships between embeddings capture musical similarity.

### About Evaluation

**Q: Why use MAP instead of simple accuracy?**

A: Accuracy measures classification ("is this song X? yes/no"). But cover song identification is a RETRIEVAL task: given a query, rank all candidates and return the best matches. MAP measures ranking quality - how well the true covers are ranked among all candidates. It considers not just whether covers are found, but how highly they are ranked.

**Q: What is a good MAP score for Indian music?**

A: It depends on dataset size and difficulty:
- MAP > 0.80: Excellent (most covers ranked at the very top)
- MAP 0.50-0.80: Good (covers generally near the top)
- MAP 0.20-0.50: Moderate (model is learning but struggles with some songs)
- MAP < 0.20: Poor (barely better than random for a small dataset)

Random baseline MAP is approximately 1/num_classes. For 245 classes: ~0.004.

**Q: Why extract embeddings instead of using classification output directly?**

A: The classification head (fc1) only works for songs seen during training. If you encounter a new song not in the training set, fc1 has no class for it. The embedding from fc0, however, is a general-purpose representation that works for ANY song, including ones never seen during training. This generalization ability is what makes the system practically useful.

### About the CQT

**Q: Why take magnitude and discard phase from the CQT?**

A: Phase contains timing information that is highly variable between recordings. Two covers of the same song will have completely different phase patterns due to differences in recording start time, tempo, and production. Magnitude tells us WHAT frequencies are present (the musical content), while phase tells us precise signal alignment (irrelevant for song identity).

**Q: Why 84 bins? Could we use more?**

A: 84 = 7 octaves x 12 bins per octave. 12 bins per octave = one bin per semitone (the smallest interval in both Western and Indian music). 7 octaves (C1 to B7) covers virtually all musical instruments. You could use more bins per octave (e.g., 24 for quarter-tone resolution, potentially useful for microtonal Indian music), but 12 has been found sufficient for cover song identification. More bins would also increase computational cost.

**Q: Why temporal pooling of 20 frames? Why not 10 or 50?**

A: 20 frames at the default CQT hop_length gives approximately 464ms per pooled frame. This is a musically meaningful timescale - roughly one beat at moderate tempo. Too few frames (e.g., 5) preserves too much temporal detail that varies between covers. Too many frames (e.g., 50) loses important rhythmic and melodic timing information. The value 20 was chosen by the original CQTNet authors empirically.

---

## 11. File Structure of the Project

```
song_identification_from_cover_song/
  CQTNet_Training_From_Scratch.ipynb   # The main notebook (run this in Google Colab)
  REFERENCE_GUIDE.md                    # This file (knowledge reference)
  dataset/                              # Your MP3 files go here (create this folder)
    song_001/
      original.mp3
      cover_1.mp3
      cover_2.mp3
    song_002/
      original.mp3
      cover_1.mp3
    ...
  cqt_features/                         # Auto-generated CQT .npy files (created by notebook)
    0_0.npy                             # Class 0, version 0
    0_1.npy                             # Class 0, version 1
    0_2.npy                             # Class 0, version 2
    1_0.npy                             # Class 1, version 0
    1_1.npy                             # Class 1, version 1
    ...
  cqtnet_indian_music.pth              # Saved model weights (created after training)
```

### File Descriptions

**CQTNet_Training_From_Scratch.ipynb**
- The main Jupyter notebook containing all code
- Divided into sections: Setup, CQT explanation, Dataset preparation, Model definition, Training, Evaluation
- Has DEMO_MODE flag: set True for synthetic data, False for real MP3 data
- Designed to run end-to-end in Google Colab

**REFERENCE_GUIDE.md (this file)**
- Complete reference documentation
- Designed to be self-contained for pasting into AI assistants
- Covers theory, implementation details, and practical guidance

**dataset/ folder**
- You create this manually with your Indian music MP3 files
- One subfolder per original song
- All versions/covers of a song go in the same subfolder
- Not included in the repository (you supply your own music)

**cqt_features/ folder**
- Auto-generated by running the notebook
- Contains .npy files (NumPy binary arrays) with preprocessed CQT spectrograms
- Each file shape: (84, T) where T depends on song length
- Named as {class_id}_{version_id}.npy

**cqtnet_indian_music.pth**
- PyTorch model checkpoint saved after training
- Contains: model weights (state_dict), optimizer state, training metadata
- Size: approximately 17 MB
- Can be loaded later for inference without retraining

---

## 12. References

### Papers

- **Original CQTNet paper**: "Learning a Representation for Cover Song Identification Using Convolutional Neural Network" by Zhesong Yu, Xiaoshuo Xu, Xiaoou Chen, and Deshun Yang. ICASSP 2020. [This paper introduces the CQTNet architecture and training methodology used in our project.]

- **Faculty's paper**: "Cover Song Identification for Indian Popular Music" - ISMIR 2025. [This paper applies CQTNet and variants to Indian music, demonstrating the value of vocal separation and triplet loss for this domain.]

### Code

- **Original CQTNet code**: https://github.com/yzspku/CQTNet [Reference implementation in PyTorch by the original paper authors]

### Datasets

- **SHS100K (Second Hand Songs 100K)**: Large-scale Western music cover song dataset with ~10,000 songs and ~100,000 recordings. Used to train the original CQTNet.

- **CoversIndian**: 245 song pairs from Hindi and Malayalam film music. One of the first systematic Indian cover song datasets.

### Libraries

- **librosa**: https://librosa.org/doc/main/generated/librosa.cqt.html [Python library for audio analysis, used for CQT computation]
- **PyTorch**: https://pytorch.org/docs/stable/index.html [Deep learning framework used for model definition, training, and inference]
- **NumPy**: https://numpy.org/doc/ [Numerical computing library for array operations]
- **scikit-learn**: https://scikit-learn.org/ [Machine learning utilities, used for evaluation metrics]

### Concepts

- **Constant-Q Transform**: Logarithmically-spaced frequency analysis. Named because the Q factor (ratio of center frequency to bandwidth) is constant across all frequency bins.
- **Cover Song Identification**: MIR task of matching different recordings of the same underlying composition.
- **Metric Learning**: The general paradigm of learning representations where distance/similarity reflects semantic relatedness. CQTNet uses classification as a proxy for metric learning.

---

*End of Reference Guide. This document covers the complete CQTNet training pipeline for Indian music cover song identification, from audio preprocessing through model architecture, training, and evaluation.*
