#!/bin/bash
# Quick script to update requirements.txt
# Usage: ./update_requirements.sh

echo "🚀 Updating requirements.txt..."
echo ""

# Check if we're in a conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  No conda environment detected!"
    echo "Please activate your environment first:"
    echo "  conda activate portal"
    exit 1
fi

echo "📦 Current environment: $CONDA_DEFAULT_ENV"
echo ""

# Run the generator
python generate_requirements.py

# Show the result
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Done! requirements.txt has been updated."
    echo ""
    echo "📝 To view changes:"
    echo "   git diff requirements.txt"
    echo ""
    echo "🧪 To test locally:"
    echo "   streamlit run Home.py"
    echo ""
    echo "🚀 To deploy:"
    echo "   git add requirements.txt"
    echo "   git commit -m 'Update requirements'"
    echo "   git push"
else
    echo ""
    echo "❌ Error: Failed to generate requirements.txt"
    exit 1
fi

