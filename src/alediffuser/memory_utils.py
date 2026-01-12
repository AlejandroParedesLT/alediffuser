import torch


def get_single_sample_memory(model, image_shape, dtype=torch.float32, device='cuda'):
    """
    Measure memory for ONE SINGLE sample (batch_size=1)
    
    Args:
        model: Your model
        image_shape: (C, H, W) - e.g., (1, 32, 32)
        dtype: torch.float32, etc.
        device: 'cuda'
    
    Returns:
        Memory in MB for a single sample
    """
    if not torch.cuda.is_available():
        print("CUDA not available")
        return None
    
    model = model.to(device)
    model.train()
    
    # Clear GPU
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    mem_before = torch.cuda.memory_allocated(device)
    
    # ONE sample only
    img = torch.randn(1, *image_shape, device=device, dtype=dtype)
    cond = torch.randint(0, 10, (1,), device=device)
    
    # Forward + backward
    pred_noise, noise = model(img, cond)
    loss = ((pred_noise - noise) ** 2).mean()
    loss.backward()
    
    mem_after = torch.cuda.max_memory_allocated(device)
    
    # Clean up
    del img, cond, pred_noise, noise, loss
    torch.cuda.empty_cache()
    
    single_sample_mb = (mem_after - mem_before) / (1024**2)
    
    return single_sample_mb


def calculate_batch_size(single_sample_mb, available_memory_gb, safety_margin=0.8):
    """
    Calculate max batch size from single sample memory
    
    Args:
        single_sample_mb: Memory for 1 sample in MB
        available_memory_gb: GPU memory in GB
        safety_margin: Use 80% of memory
    
    Returns:
        Max batch size
    """
    available_mb = available_memory_gb * 1024 * safety_margin
    max_batch = int(available_mb / single_sample_mb)
    return max_batch

