from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.host_specific_recovery.io.common import transpose_numeric


DEFAULT_KUROKAWA_DIR = Path(
    r"C:\Users\USER\OneDrive\Desktop\Antibiotics\Kurokawa_et_al"
)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    column_sums = df.sum(axis=0)
    return df.div(column_sums.where(column_sums != 0), axis=1).fillna(0.0)


def _load_mouse_tables(data_dir: Path, mouse_id: str, normalize: bool) -> Dict[str, object]:
    relative_abundance_path = data_dir / f"{mouse_id}_relative_abundance.csv"
    taxonomy_path = data_dir / f"{mouse_id}_taxonomy.csv"

    relative_abundance = pd.read_csv(relative_abundance_path, index_col=0)
    taxonomy = pd.read_csv(taxonomy_path, index_col="MAG_core")

    relative_abundance = relative_abundance.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if normalize:
        relative_abundance = _normalize_columns(relative_abundance)

    missing_taxonomy = relative_abundance.index.difference(taxonomy.index)
    if len(missing_taxonomy) > 0:
        raise ValueError(
            f"{mouse_id}: {len(missing_taxonomy)} taxa in relative abundance are missing from taxonomy."
        )

    taxonomy = taxonomy.loc[relative_abundance.index]

    return {
        "relative_abundance_df": relative_abundance,
        "taxonomy_df": taxonomy,
        "relative_abundance": transpose_numeric(relative_abundance, norm=False),
        "sample_ids": list(relative_abundance.columns),
        "taxa_ids": list(relative_abundance.index),
    }


def load_Kurokawa_et_al_data(
        dir: Optional[str | Path] = None,
        normalize: bool = False,
) -> Dict[str, object]:
    data_dir = Path(dir) if dir is not None else DEFAULT_KUROKAWA_DIR

    mouse_c = _load_mouse_tables(data_dir, "mouseC", normalize=normalize)
    mouse_d = _load_mouse_tables(data_dir, "mouseD", normalize=normalize)

    return {
        "mouseC": mouse_c,
        "mouseD": mouse_d,
        "mouseC_relative_abundance_df": mouse_c["relative_abundance_df"],
        "mouseC_taxonomy_df": mouse_c["taxonomy_df"],
        "mouseC_relative_abundance": mouse_c["relative_abundance"],
        "mouseC_sample_ids": mouse_c["sample_ids"],
        "mouseC_taxa_ids": mouse_c["taxa_ids"],
        "mouseD_relative_abundance_df": mouse_d["relative_abundance_df"],
        "mouseD_taxonomy_df": mouse_d["taxonomy_df"],
        "mouseD_relative_abundance": mouse_d["relative_abundance"],
        "mouseD_sample_ids": mouse_d["sample_ids"],
        "mouseD_taxa_ids": mouse_d["taxa_ids"],
    }
