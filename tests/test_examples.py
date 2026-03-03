import subprocess
import os

def test_example_script_exists():
    """Test that example script exists and is executable"""
    assert os.path.exists('examples/vectorize_image.py')
