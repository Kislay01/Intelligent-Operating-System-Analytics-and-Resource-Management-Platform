import os
import tempfile

# Keep test runs from writing into the project's data/ directory.
os.environ.setdefault("KS_DATA_DIR", tempfile.mkdtemp(prefix="kernelsense-test-"))
