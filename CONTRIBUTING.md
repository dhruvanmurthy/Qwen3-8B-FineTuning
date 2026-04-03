# Contributing Guide

Welcome to the Qwen3-8B Fine-tuning project! This guide explains how to contribute.

## Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/dhruvanmurthy/Qwen3-8B-FineTuning.git
   cd Qwen3-8B-FineTuning
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Set up development environment**
   ```bash
   pip install -r requirements.txt
   ```

## Areas for Contribution

### 1. New Datasets
Add support for more tool-use datasets:
- [ ] Markbench
- [ ] APIBench
- [ ] Custom domain datasets

**How to contribute**:
1. Add dataset loading logic in `src/data_loader.py`
2. Update `configs/dataset_config.yaml`
3. Add example in `DATASET_CARD.md`
4. Create PR with test dataset

### 2. New Evaluation Metrics
Improve evaluation beyond tool selection accuracy:
- [ ] Semantic similarity of generated calls
- [ ] Parameter type correctness
- [ ] Hallucination detection rate
- [ ] Robustness to tool variations

**How to contribute**:
1. Add metric to `src/evaluate.py`
2. Document in `docs/EVALUATION.md`
3. Add benchmark results

### 3. Multi-Node Distributed Training (Future)
Tinker handles remote GPU training. For local multi-GPU support:
- [ ] DeepSpeed ZeRO Stage 2/3 integration
- [ ] FSDP support

**How to contribute**:
1. Implement in `src/train.py`
2. Test on multi-GPU setup
3. Document speedup results

### 4. Training Improvements
Further optimize training:
- [ ] Mixed precision training (FP8)
- [ ] Quantization-aware training
- [ ] Knowledge distillation to smaller models
- [ ] Data augmentation strategies

**How to contribute**:
1. Implement training improvement
2. Compare training time and quality
3. Show final model quality metrics

### 5. Inference Optimization
Improve model serving:
- [ ] ONNX export
- [ ] TensorRT optimization
- [ ] Batched inference
- [ ] Quantized inference

**How to contribute**:
1. Add optimization in new file
2. Benchmark latency/throughput
3. Document usage examples
4. Test on different hardware

### 6. Documentation & Tutorials
Improve project documentation:
- [ ] Video tutorial
- [ ] Jupyter notebook walkthrough
- [ ] Deployment guide (HF Spaces, Replicate)
- [ ] Multi-lingual documentation

**How to contribute**:
1. Create clear, reproducible content
2. Include code examples
3. Test all instructions
4. Submit PR with content

## Development Workflow

### 1. Code Quality

**Format code**:
```bash
black src/
isort src/
```

**Lint**:
```bash
flake8 src/
mypy src/
```

**Run tests**:
```bash
pytest tests/
```

### 2. Commit Messages

Use conventional commits:
```
feat: add new evaluation metric (argument correctness)
fix: resolve GPU memory leak in training loop
docs: update TRAINING_PLAN.md with new results
refactor: simplify data loading pipeline
test: add unit tests for tokenization
```

### 3. Pull Request Process

1. **Before submitting**:
   ```bash
   # Format
   black src/
   isort src/

   # Lint
   flake8 src/

   # Test (if applicable)
   pytest tests/
   ```

2. **Write PR description**:
   - What does this change?
   - Why is it needed?
   - How does it work?
   - Results/benchmarks (if applicable)

3. **Target `main` branch**

4. **Wait for review**
   - Author will review within 48 hours
   - Address feedback
   - Re-request review

### 4. Testing

For new features, add tests in `tests/` directory:

```python
# tests/test_data_loader.py
def test_load_datasets():
    loader = ToolUseDataLoader()
    datasets = loader.load_all_datasets()
    assert len(datasets) > 0
    assert "source" in datasets.column_names
```

Run tests:
```bash
pytest tests/ -v
```

## Documentation Standards

### Code Comments
```python
def extract_tool_name(text: str) -> str:
    """Extract tool name from generated text.

    Args:
        text: Generated text containing tool calls

    Returns:
        Tool name extracted from text, or None if not found

    Example:
        >>> extract_tool_name('Use get_weather(city="Paris")')
        'get_weather'
    """
```

### Docstrings
Use Google-style docstrings:
```python
def train(model_args, data_args, training_args):
    """Main training function.

    Loads model, prepares data, and runs training loop.

    Args:
        model_args: Model configuration
        data_args: Data configuration
        training_args: Training hyperparameters

    Returns:
        TrainingResult object with metrics

    Raises:
        ValueError: If required arguments missing
        RuntimeError: If GPU/CUDA not available
    """
```

### Markdown Files
- Use clear headings and hierarchy
- Include code examples
- Add links to relevant sections
- Use tables for comparisons
- Include warnings/notes for important info

## Reporting Issues

**Before reporting**:
1. Search existing issues
2. Check TROUBLESHOOTING.md
3. Test with latest code

**When reporting**:
1. Provide error message and stack trace
2. Include minimal reproducible example
3. Share system info:
   ```bash
   python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
   nvidia-smi
   ```
4. Describe expected vs actual behavior

## Feature Requests

Provide:
1. Clear description of wanted feature
2. Use cases/motivation
3. How it benefits project
4. Suggested implementation (optional)
5. Examples of similar projects (optional)

## Performance Benchmarks

When optimizing, provide before/after:
```markdown
**Benchmark**: Multi-GPU Training (2x A100)

Before:
- Training time: 24 hours
- Cost: $72
- GPU utilization: 85%

After:
- Training time: 18 hours
- Cost: $54
- GPU utilization: 92%

Improvement: -25% training time, -25% cost
```

## Becoming a Maintainer

Requirements:
- 5+ merged PRs
- High quality code
- Active engagement
- Understanding of project goals

Contact project lead: @dhruvanmurthy

## Code of Conduct

- Be respectful and inclusive
- No harassment or discrimination
- Constructive feedback only
- Assume good intent

## Questions?

- **Technical**: Open GitHub discussion
- **Contribution**: Reply to issue
- **Maintainer**: Email @dhruvanmurthy

---

Thank you for contributing! 🎉
