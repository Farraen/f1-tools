#!/usr/bin/env python3
"""
Generate requirements.txt from current conda environment
Usage: python generate_requirements.py
"""

import subprocess
from datetime import datetime

# Define essential packages for your Streamlit app
# Only top-level packages - pip will handle dependencies
ESSENTIAL_PACKAGES = [
    'altair', 'annotated-types', 'ansi2html', 'antlr4-python3-runtime', 'anyio',
    'attrs', 'beautifulsoup4', 'blinker', 'cachetools', 'catboost', 'click',
    'cloudpickle', 'colourmap', 'contourpy', 'd3graph', 'daal4py', 'daal',
    'dacite', 'dash-core-components', 'dash-html-components', 'dash-table', 'dash',
    'datazets', 'distro', 'diversipy', 'dnspython', 'doepy', 'entrypoints',
    'et-xmlfile', 'exceptiongroup', 'faker', 'favicon', 'flask', 'fmpy',
    'gitdb', 'gitpython', 'h11', 'htbuilder', 'htmlmin', 'httpcore', 'httpx',
    'imagehash', 'imageio', 'importlib-metadata', 'ismember', 'itsdangerous',
    'jinja2', 'jiter', 'joblib', 'jsonschema-specifications', 'jsonschema',
    'kiwisolver', 'lark', 'lazy-loader', 'llvmlite', 'lxml', 'markdown-it-py',
    'markdown', 'markdownlit', 'markupsafe', 'matplotlib-inline', 'matplotlib',
    'mdurl', 'more-itertools', 'msgpack', 'multimethod', 'nest-asyncio',
    'numba', 'numpy', 'omegaconf', 'openai', 'openpyxl', 'orjson', 'packaging',
    'pandas-dq', 'pandas-profiling', 'pandas', 'password-strength', 'patsy',
    'phik', 'pillow', 'plotly-resampler', 'plotly', 'protobuf', 'pyaml',
    'pyarrow', 'pydantic-core', 'pydantic', 'pydeck', 'pydoe', 'pygad',
    'pygments', 'pymdown-extensions', 'pymongo', 'pyro-ppl', 'python-dotenv',
    'python-louvain', 'pywavelets', 'pyyaml', 'referencing', 'retrying', 'rich',
    'rpds-py', 'scikit-base', 'scikit-image', 'scikit-learn-intelex',
    'scikit-learn', 'scikit-optimize', 'scikit-plot', 'seaborn', 'smmap',
    'sniffio', 'soupsieve', 'speechrecognition', 'sseclient-py', 'st-annotated-text',
    'statsmodels', 'streamlit-camera-input-live', 'streamlit-card',
    'streamlit-d3graph', 'streamlit-embedcode', 'streamlit-extras',
    'streamlit-faker', 'streamlit-image-coordinates', 'streamlit-keyup',
    'streamlit-mic-recorder', 'streamlit-toggle-switch', 'streamlit-vertical-slider',
    'streamlit', 'tabpfn-client', 'tangled-up-in-unicode', 'tbb', 'threadpoolctl',
    'tifffile', 'toml', 'toolz', 'torch', 'torchvision', 'tornado', 'tqdm',
    'trace-updater', 'traitlets', 'tsdownsample', 'typeguard', 'typing-extensions',
    'visions', 'werkzeug', 'wordcloud', 'xxhash', 'ydata-profiling'
]

def get_installed_packages():
    """Get all installed packages from pip"""
    print("📦 Reading installed packages...")
    result = subprocess.run(['pip', 'list', '--format=freeze'], 
                          capture_output=True, text=True, check=True)
    
    installed = {}
    for line in result.stdout.split('\n'):
        if '==' in line:
            pkg, ver = line.strip().split('==', 1)
            # Store multiple name variations for matching
            installed[pkg.lower()] = ver
            installed[pkg.lower().replace('-', '_')] = ver
            installed[pkg.lower().replace('_', '-')] = ver
    
    return installed

def normalize_package_name(pkg_name):
    """Normalize package name for matching"""
    return pkg_name.lower().replace('-', '_').replace('_', '-')

def generate_requirements(output_file='requirements.txt'):
    """Generate requirements.txt from essential packages"""
    
    print("=" * 80)
    print("🚀 GENERATING REQUIREMENTS.TXT")
    print("=" * 80)
    
    # Get installed packages
    installed = get_installed_packages()
    
    # Match and write requirements
    found_packages = []
    missing_packages = []
    
    for pkg in sorted(ESSENTIAL_PACKAGES):
        # Try to find package with various name formats
        version = None
        for variant in [pkg.lower(), pkg.lower().replace('-', '_'), 
                       pkg.lower().replace('_', '-')]:
            if variant in installed:
                version = installed[variant]
                break
        
        if version:
            found_packages.append((pkg, version))
        else:
            missing_packages.append(pkg)
            print(f"⚠️  Warning: {pkg} not found in environment")
    
    # Write requirements.txt
    with open(output_file, 'w') as f:
        f.write('# Requirements for Streamlit Cloud deployment\n')
        f.write(f'# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('# Curated list of top-level packages\n')
        f.write('# Run: python generate_requirements.py to regenerate\n\n')
        
        for pkg, version in found_packages:
            f.write(f'{pkg}=={version}\n')
    
    # Print summary
    print("\n" + "=" * 80)
    print("✅ SUMMARY")
    print("=" * 80)
    print(f"  Found packages:   {len(found_packages)}")
    print(f"  Missing packages: {len(missing_packages)}")
    print(f"  Output file:      {output_file}")
    
    if missing_packages:
        print("\n⚠️  MISSING PACKAGES:")
        print("=" * 80)
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nTo install missing packages:")
        print(f"  pip install {' '.join(missing_packages)}")
    
    print("\n" + "=" * 80)
    print("✅ requirements.txt generated successfully!")
    print("=" * 80)
    
    # Display key packages
    key_packages = ['pandas', 'torch', 'streamlit', 'catboost', 'pydantic']
    print("\n📌 Key package versions:")
    print("-" * 80)
    for pkg, version in found_packages:
        if pkg.lower() in key_packages:
            print(f"  {pkg:20} {version}")
    
    print("\n💡 Next steps:")
    print("-" * 80)
    print("  1. Review requirements.txt")
    print("  2. Test locally: streamlit run Home.py")
    print("  3. Commit and push to deploy")
    print("=" * 80)

if __name__ == '__main__':
    try:
        generate_requirements()
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: Failed to get installed packages: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)

