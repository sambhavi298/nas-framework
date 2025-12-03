# Running on Google Colab (T4 GPU)

Follow these steps to train the final model on Google Colab using a T4 GPU.

### 1. Setup Colab Environment
1. Open [Google Colab](https://colab.research.google.com/).
2. Create a **New Notebook**.
3. Go to **Runtime** > **Change runtime type**.
4. Select **T4 GPU** under Hardware accelerator and click **Save**.

### 2. Run the Code
Copy and paste the following code block into a code cell and run it:

```python
# 1. Clone the repository
!git clone https://github.com/sambhavi298/nas-framework.git

# 2. Enter the project directory
%cd nas-framework

# 3. Install dependencies
!pip install -r requirements.txt

# 4. Run the training script
# The script automatically detects the GPU
!python scripts/train_final.py
```

### 3. Run the Search (Optional)
If you want to discover a new architecture instead of training the existing one:

```python
# Run the search script
!python scripts/search.py
```

### 4. Expected Output
- **Search**: Will run for 30 epochs and save genotypes to `search_runs/`.
- **Train**: Will run for 100 epochs and print the final test accuracy.
