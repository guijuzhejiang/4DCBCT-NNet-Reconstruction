import os
os.environ["HF_HUB_HTTP"] = "http://127.0.0.1:1095"
os.environ["HF_HUB_HTTPS"] = "http://127.0.0.1:1095"

from monai.bundle import download

download(name="spleen_ct_segmentation", bundle_dir="./bundles/")
