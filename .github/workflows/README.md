# GitHub Actions for sLAM

## Test sLAM Model Generation

This workflow (`test-slam-model.yml`) automatically tests that the sLAM model generation pipeline works correctly.

### What it does

1. **Sets up environment**: Installs Python 3.9 and all required dependencies
2. **Downloads NLTK data**: Fetches required tokenization data
3. **Runs model training**: Executes `make-slam.py` with minimal test parameters:
   - 10 datasets (instead of thousands)
   - 1 epoch (instead of 3+)
   - Smaller model dimensions (context_size=16, d_model=64)
4. **Verifies outputs**: Checks that both required files are created:
   - `test-model.keras` - The trained model file
   - `test-model.pkl` - The tokenizer/encoder file
5. **Tests generation**: Runs `generate.py` to ensure the model can generate text
6. **Uploads artifacts**: Saves the generated files for inspection (retained for 7 days)

### When it runs

- On every push to `main` or `master` branch
- On every pull request to `main` or `master` branch
- Manually via the "Actions" tab in GitHub (workflow_dispatch)

### Expected runtime

The test should complete in approximately 5-10 minutes on GitHub's runners.

### Testing locally

Before pushing changes, you can test the workflow locally:

```bash
# From the project root directory
./test-ci-locally.sh
```

This script mimics the GitHub Action steps and will verify everything works before you push.

### Configuration

If you need to adjust the test parameters, edit the "Run make-slam.py" step in `test-slam-model.yml`:

- `--num_datasets`: Number of cc_news datasets to use (default: 10)
- `--epochs`: Number of training epochs (default: 1)
- `--context_size`: Context window size (default: 16)
- `--d_model`: Embedding dimensions (default: 64)

**Note**: Keep these values small to ensure fast CI runs. The goal is to verify the pipeline works, not to create a high-quality model.

### Troubleshooting

If the workflow fails:

1. **Check the logs** in the GitHub Actions tab
2. **Run locally** using `./test-ci-locally.sh` to reproduce the issue
3. **Common issues**:
   - Memory errors: Reduce `--num_datasets` or `--d_model`
   - Timeout: Reduce `--epochs` or increase `timeout-minutes` in the workflow
   - Missing files: Check that the `--name` parameter matches the verification steps

### Files created

The workflow creates these files during the test:

- `test-model.keras` - Keras model file (~few MB with test parameters)
- `test-model.pkl` - Tokenizer pickle file (~KB)
- `token_number_distribution.png` - Histogram (if verbose mode)
- `chunk_length_distribution.png` - Histogram (if verbose mode)

**Note**: All test artifacts are automatically cleaned up after the workflow completes. The files are uploaded as artifacts (retained for 7 days) before cleanup, so you can download them for inspection if needed.
