from ocr import column_extractor_scc
import definitions
import argparse
import os



column_extractor_scc.run_columns(os.path.join(definitions.ROOT_DIR, "data"))