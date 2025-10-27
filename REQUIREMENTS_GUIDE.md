# Requirements.txt Generator Guide

## Quick Start

Generate a fresh `requirements.txt` from your current conda environment:

```bash
# Activate your environment first
conda activate portal

# Run the generator script
python generate_requirements.py
```

That's it! Your `requirements.txt` will be updated.

## What This Script Does

1. **Reads installed packages** from your current environment
2. **Matches only essential packages** (150 top-level packages)
3. **Generates clean requirements.txt** with proper formatting
4. **Shows summary** of found/missing packages
5. **Displays key versions** (pandas, torch, streamlit, etc.)

## When to Use This

Run this script whenever:
- ✅ You update packages in your conda environment
- ✅ Before deploying to Streamlit Cloud
- ✅ After installing new dependencies
- ✅ When requirements.txt gets out of sync

## Customizing the Package List

To add or remove packages from requirements.txt:

1. Open `generate_requirements.py`
2. Find the `ESSENTIAL_PACKAGES` list
3. Add or remove package names
4. Run the script again

Example:
```python
ESSENTIAL_PACKAGES = [
    'streamlit',
    'pandas',
    'numpy',
    # Add your packages here
    'my-new-package',
]
```

## Output Format

The generated `requirements.txt` will have:
- Clean header with generation timestamp
- Alphabetically sorted packages
- Exact version pins (e.g., `pandas==2.3.1`)
- No dependency packages (only top-level)

## Troubleshooting

### Package Not Found Warning

If you see:
```
⚠️  Warning: my-package not found in environment
```

**Solution:**
```bash
pip install my-package
python generate_requirements.py
```

### Wrong Versions

If package versions don't match what you expect:

**Check installed version:**
```bash
pip show package-name
```

**Update to specific version:**
```bash
pip install package-name==1.2.3
python generate_requirements.py
```

### Environment Mismatch

Always make sure you're in the correct conda environment:

```bash
# Check current environment
conda info --envs

# Activate the right one
conda activate portal

# Verify with
which python
```

## Best Practices

1. **Run before every deployment** to ensure sync
2. **Test locally first** after regenerating
3. **Commit both files** if you modify `generate_requirements.py`
4. **Keep backups** of working `requirements.txt` files

## Example Workflow

```bash
# 1. Update your environment
conda activate portal
pip install new-package==1.0.0

# 2. Regenerate requirements.txt
python generate_requirements.py

# 3. Test locally
streamlit run Home.py

# 4. Commit and deploy
git add requirements.txt
git commit -m "Update requirements"
git push
```

## Key Package Versions (Current)

The script will automatically show your current versions of:
- `pandas` (important: v2.x has breaking changes)
- `torch` (PyTorch for ML)
- `streamlit` (app framework)
- `catboost` (ML library)
- `pydantic` (important: v2.x has breaking changes)

## Notes

- The script only includes **top-level packages**
- Dependencies are automatically installed by pip
- This keeps requirements.txt **clean and maintainable**
- **Fewer pins = more flexible = less fragile** on Streamlit Cloud

