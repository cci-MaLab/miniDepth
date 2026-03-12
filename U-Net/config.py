# ---- CONFIG ----

BASE_SEED=12345         # for reproducibility 
# Patch sampling
PATCH_SIZE = 256
POS_MIN_FRAC = 0.005   # 0.5% minimum positive pixels for positive-biased patches
RAND_MAX_FRAC = 0.02   # 2% maximum positive pixels for random patches
DILATE_RADIUS = 7      # for candidate map

# Sampling ratios
WARMUP_POS_RATIO = 0.90   # warmup phase (pos-biased fraction)
LATER_POS_RATIO = 0.90    # later phase (pos-biased fraction)

# Training
PATCHES_PER_PLANE_PER_EPOCH = 100
VAL_PATCHES_PER_PLANE = 80


# Validation
VAL_FIXED_RATIO = 0.90   # 50/50 pos:rand for fixed-val

# Number of epochs for training
NUM_EPOCHS = 35
WARM_UP_EPOCHS = 10

# Device
DEVICE = "cuda"

# Early stopping
PATIENCE_LIMIT = 5

EPS=1e-6

#LOSS = "Dice+BCE"
LOSS = "Tversky"
#LOSS = "Focal-Tversky"
#LOSS = "IOU"

#Tversky loss
ALPHA=0.3
BETA=0.7
#Focal Tversky loss
GAMMA=1.33