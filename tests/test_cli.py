import json
import subprocess
import sys
from glob import glob
import os
import pytest

images = glob(os.path.join(os.path.dirname(__file__), "..", "samples", "*"))

@pytest.mark.parametrize("img", images)
def test_count_cli_outputs_json_and_nonnegative(img):
    # Run the CLI forcing CPU to keep CI friendly
    cmd = [sys.executable, "-m", "dfine_count", "count", "--image", img, "--device", "cpu"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"CLI failed: {proc.stderr}"
    data = json.loads(proc.stdout)
    assert "total_count" in data
    assert isinstance(data["total_count"], int)
    assert data["total_count"] >= 0
