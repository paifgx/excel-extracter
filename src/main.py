import pandas as pd
import numpy as np
import re
import logging
from typing import Tuple, List, Optional, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def is_candidate_header(
    row: pd.Series,
    n_cols: int,
    fill_ratio_threshold: float,
    text_ratio_threshold: float,
    known_header_keywords: Optional[List[str]] = None,
) -> bool:
    """Check if a row is a candidate for a header."""
    fill_ratio = row.count() / n_cols
    text_ratio = sum(isinstance(x, str) for x in row) / n_cols
    candidate = (
        fill_ratio >= fill_ratio_threshold and text_ratio >= text_ratio_threshold
    )
    
    if candidate and known_header_keywords:
        row_text = " ".join(str(x).lower() for x in row if isinstance(x, str))
        candidate = any(kw.lower() in row_text for kw in known_header_keywords)
    
    return candidate


def detect_header_indices(
    df_raw: pd.DataFrame,
    fill_ratio_threshold: float,
    text_ratio_threshold: float,
    max_header_rows: int,
    known_header_keywords: Optional[List[str]] = None,
) -> List[int]:
    """Find header candidates in the first rows and return their indices."""
    n_cols = df_raw.shape[1]
    candidate_indices = []
    
    for i in range(min(max_header_rows, len(df_raw))):
        row = df_raw.iloc[i]
        if is_candidate_header(
            row,
            n_cols,
            fill_ratio_threshold,
            text_ratio_threshold,
            known_header_keywords,
        ):
            candidate_indices.append(i)
    
    if not candidate_indices:
        # Fallback: search in first 10 rows
        for i in range(min(10, len(df_raw))):
            row = df_raw.iloc[i]
            if is_candidate_header(
                row,
                n_cols,
                fill_ratio_threshold,
                text_ratio_threshold,
                known_header_keywords,
            ):
                candidate_indices.append(i)
                break
    
    if not candidate_indices:
        raise ValueError("No header found based on heuristic.")
    
    return candidate_indices


def combine_headers(
    df_raw: pd.DataFrame, header_indices: List[int], valid_cols: List[int]
) -> List[str]:
    """Combine header rows into final column names and handle duplicates."""
    n_cols = df_raw.shape[1]
    final_header: List[Optional[str]] = []
    
    for col in range(n_cols):
        header_parts: List[str] = []
        for idx in header_indices:
            cell = df_raw.iloc[idx, col]
            if pd.isna(cell):
                header_parts.append("")
            else:
                cleaned = re.sub(r"\s+", " ", str(cell)).strip()
                header_parts.append(cleaned)
        
        combined = " ".join(part for part in header_parts if part)
        final_header.append(combined if combined else None)
    
    # Keep only columns with content
    filtered_header: List[str] = []
    for i in valid_cols:
        header_value = final_header[i]
        if header_value is not None:
            filtered_header.append(header_value)
    
    # Handle duplicates
    seen: Dict[str, int] = {}
    for i, name in enumerate(filtered_header):
        if name in seen:
            seen[name] += 1
            filtered_header[i] = f"{name}_{seen[name]}"
        else:
            seen[name] = 1

    return filtered_header


def extract_data_rows(
    df_raw: pd.DataFrame, header_indices: List[int], valid_cols: List[int]
) -> pd.DataFrame:
    """Extract data rows after the last header."""
    data_start_idx = header_indices[-1] + 1
    df_data = df_raw.iloc[data_start_idx:, valid_cols].copy()
    df_data["orig_index"] = df_raw.index[data_start_idx:]
    return df_data


def clean_dataframe(df_data: pd.DataFrame, header: List[str]) -> pd.DataFrame:
    """Clean cells by removing extra whitespace and replacing empty strings with NaN."""
    df_data = df_data.replace(r"^\s*$", np.nan, regex=True).infer_objects(copy=False)
    df_data = df_data.dropna(subset=header, how="all")

    def clean_cell(x):
        if isinstance(x, str):
            return re.sub(r"\s+", " ", x.replace("\n", " ")).strip()
        return x

    df_data[header] = df_data[header].apply(lambda col: col.map(clean_cell))
    return df_data


def adaptive_filtering(
    df_data: pd.DataFrame, header: List[str], fill_ratio_threshold: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter rows based on adaptive fill ratio."""
    fill_ratios = df_data[header].notna().sum(axis=1) / len(header)
    median_fill = fill_ratios.median()
    threshold = fill_ratio_threshold * median_fill
    
    valid_mask = fill_ratios >= threshold
    
    valid_df = df_data[valid_mask].reset_index(drop=True)
    removed_df = df_data[~valid_mask].reset_index(drop=True)
    
    return valid_df, removed_df


def extract_table_from_sheet(
    df_raw: pd.DataFrame,
    fill_ratio_threshold: float = 0.5,
    text_ratio_threshold: float = 0.8,
    max_header_rows: int = 2,
    auto_detect_multiline: bool = False,
    known_header_keywords: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract a table from a DataFrame loaded from an Excel sheet.
    
    Returns:
        Tuple(valid_data, removed_data)
    """
    n_cols = df_raw.shape[1]
    logging.info("Searching for header indices.")
    header_indices = detect_header_indices(
        df_raw,
        fill_ratio_threshold,
        text_ratio_threshold,
        max_header_rows,
        known_header_keywords,
    )
    logging.info(f"Found header indices: {header_indices}")

    # Extend multiline header detection
    if auto_detect_multiline:
        for j in range(
            header_indices[-1] + 1, header_indices[-1] + max_header_rows + 1
        ):
            if j >= len(df_raw):
                break
            row = df_raw.iloc[j]
            if row.count() >= (n_cols / 2):
                header_indices.append(j)
            else:
                break
                
    header_indices = sorted(header_indices)
    logging.info(f"Finalized header indices: {header_indices}")

    # Determine valid columns (at least one valid header entry)
    final_header_full = []
    for col in range(n_cols):
        header_parts = []
        for idx in header_indices:
            cell = df_raw.iloc[idx, col]
            if pd.isna(cell):
                header_parts.append("")
            else:
                header_parts.append(re.sub(r"\s+", " ", str(cell)).strip())
                
        combined = " ".join([part for part in header_parts if part])
        final_header_full.append(combined if combined else None)
        
    valid_cols = [i for i, h in enumerate(final_header_full) if h not in (None, "")]
    if not valid_cols:
        raise ValueError("No valid column headers found.")
        
    logging.info(f"Valid columns: {valid_cols}")

    final_header = combine_headers(df_raw, header_indices, valid_cols)
    logging.info(f"Combined headers: {final_header}")

    df_data = extract_data_rows(df_raw, header_indices, valid_cols)
    df_data.columns = final_header + ["orig_index"]

    df_data = clean_dataframe(df_data, final_header)
    valid_df, removed_df = adaptive_filtering(
        df_data, final_header, fill_ratio_threshold
    )

    return valid_df, removed_df


def extract_tables_from_excel(
    file_path: str, sheet_names: Optional[List[str]] = None, **kwargs
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Extract tables from an Excel file.
    
    If multiple sheets are present, each sheet is processed individually.
    
    Returns:
        Dict with sheet names as keys and Tuple(valid_data, removed_data) as values.
    """
    logging.info(f"Reading Excel file: {file_path}")
    excel_data = pd.read_excel(file_path, sheet_name=sheet_names, header=None)
    tables = {}
    
    if isinstance(excel_data, dict):
        for sheet, df in excel_data.items():
            try:
                logging.info(f"Processing sheet: {sheet}")
                valid_df, removed_df = extract_table_from_sheet(df, **kwargs)
                tables[sheet] = (valid_df, removed_df)
            except Exception as e:
                logging.error(f"Error processing sheet '{sheet}': {e}")
    else:
        try:
            logging.info("Processing single-sheet file.")
            valid_df, removed_df = extract_table_from_sheet(excel_data, **kwargs)
            tables["Sheet1"] = (valid_df, removed_df)
        except Exception as e:
            logging.error(f"Error processing Excel file: {e}")
            
    return tables


if __name__ == "__main__":
    file_path = "data/test-2.xlsx"

    known_keywords = ["Name", "Date", "Amount"]
    result = extract_tables_from_excel(
        file_path,
        known_header_keywords=known_keywords,
        auto_detect_multiline=True,
        fill_ratio_threshold=0.5,
        text_ratio_threshold=0.8,
        max_header_rows=1,
    )

    for sheet, (valid_df, removed_df) in result.items():
        print(f"\nSheet: {sheet}")
        print("Valid Data:")
        print(valid_df.head())
        print("Removed Data:")
        print(removed_df.head())
