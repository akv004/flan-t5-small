The following changes were made in your working code to resolve training issues on your system. Below is a documentation of what was changed and why:

### Learning Rate Adjustment
- **Changed to:** `learning_rate=5e-6`
- **Why:** A lower learning rate was used to stabilize training and prevent overshooting the optimal weights. This was likely necessary due to sensitivity in gradients or the dataset characteristics.

### FP16 (Mixed Precision) Disabled
- **Changed to:** `fp16=False`
- **Why:** Disabling FP16 ensured numerical stability during training. Mixed precision can sometimes lead to gradient underflow or instability on certain hardware setups or models.

### Batch Size and Gradient Accumulation
- **Kept:** `per_device_train_batch_size=4` and `gradient_accumulation_steps=1`
- **Why:** A smaller batch size was used to fit within the GPU memory constraints, and accumulation was avoided to keep gradients directly applied without splitting them over steps.

### Gradient Clipping Enabled
- **Kept:** `max_grad_norm=1.0`
- **Why:** Clipping gradients ensures that large gradients do not destabilize training by exploding, especially when FP16 is disabled.

### Callbacks for Gradient Monitoring
- **Added:** `GradientNormLogger` and `GradientDebugger`
- **Why:** These custom callbacks were introduced to monitor and debug gradient norms, helping to ensure that gradients stay in a healthy range.

### Dataset Preprocessing
- **Preserved:** Custom preprocessing and validation split
- **Why:** Preprocessing ensures tokenized data is well-formatted, and splitting 10% as validation helps monitor overfitting.

## Summary of Key Parameters
- **Learning Rate:** Stabilized training
- **FP16 Disabled:** Ensured numerical stability
- **Batch Size:** Adjusted to hardware limits
- **Gradient Monitoring:** Debugging tool for identifying issues

These settings should be documented in the training script as comments for clarity and to help replicate this setup with other datasets.