# EDA summary: thermal fault detection

- Total images: 2,000
- Fault rate: 12.5%

## Naive brightest-pixel threshold
- Threshold set to catch 75% of faults: peak temperature > 44.9
- Resulting precision: 25.6%
- Resulting recall: 74.8%
- False positives: 544 healthy images incorrectly flagged

A single brightness threshold either misses mild faults or flags a large share of healthy images with a benign warm spot. The rest of this project compares two learned approaches against this baseline.
