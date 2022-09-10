from tdw_physics.util import MODEL_LIBRARIES
from tdw.tdw_utils import TDWUtils



def download():

    TDWUtils.download_asset_bundles(
        path="/ccn2/u/honglinc/tdw_local_asset_bundles",
        models={
            "models_special.json": [r.name for r in MODEL_LIBRARIES['models_special.json'].records],
            "models_full.json": [r.name for r in MODEL_LIBRARIES['models_full.json'].records]
        }
    )


download()
