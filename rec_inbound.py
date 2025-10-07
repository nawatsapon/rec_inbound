import pandas as pd
import os
import re
from datetime import datetime, timedelta
from typing import Dict
import logging
import numpy as np
import yaml
import sys
from sftp_manager import SFTPManager
from date_converter import DateConverter
import shutil

# Configure logging to output to both file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/wsol/app/logs/new_auto_reconcile/rec_inbound.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def load_config(
    config_file: str = "/wsol/app/new_auto_reconcile/inbound/inbound_services.yaml",
) -> Dict:
    ##def load_config(config_file: str = 'inbound_services.yaml') -> Dict:
    """Load configuration from a YAML file."""
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_file}.")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

##remove class DateTransformer cause we already changed to new DataConvertor

class BaseProcessor:
    """Base class for file processors with common utility methods."""
    def __init__(self, file_path: str, date_pattern: str):
        self.file_path = file_path
        self.date_pattern = date_pattern    

    def _get_date_from_filename(self) -> str:
        """Extracts date from filename using regex."""
        match = re.search(self.date_pattern, self.file_path)
        if match:
            return match.group(1)
        return ""

    def save_as_csv(self, df: pd.DataFrame, output_path: str):
        """Saves DataFrame to CSV."""
        try:
            df.to_csv(output_path, index=False, encoding="utf-8")
            logger.info(f"Successfully saved processed file to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save file {output_path}: {e}")
            raise

    @staticmethod
    def _create_rccpkey(
        df: pd.DataFrame, ref1_col: str, ref2_col: str, suffix: str
    ) -> pd.DataFrame:
        """Create a unique reconciliation key."""
        df["rccpkey"] = (
            df[ref1_col].astype(str) + "|" + df[ref2_col].astype(str) + "|" + suffix
        )
        return df

    @staticmethod
    def _save_csv(df: pd.DataFrame, file_path: str) -> None:
        """Save DataFrame to CSV with UTF-8 encoding."""
        df.to_csv(file_path, index=False, encoding="utf-8")
        logger.info(f"Saved {len(df)} rows to {file_path}")

    @staticmethod
    def _convert_amount_value(amount):
        """
        Convert a single amount value to a standardized decimal format as a string with two decimals.
        The conversion removes any non-numeric (and non-decimal point) characters.
        """
        # Remove any characters that are not digits or a decimal point
        s_clean = re.sub(r"[^\d.]", "", str(amount))

        # If cleaning leaves an empty string, return "0.00"
        if not s_clean:
            return "0.00"

        # Split into integer and decimal parts if a decimal point exists
        if "." in s_clean:
            integer, decimal = s_clean.split(".", 1)
            integer = integer.lstrip("0") or "0"
            # Ensure decimal part is exactly two digits (pad or truncate)
            decimal = decimal.ljust(2, "0")[:2]
        else:
            integer = s_clean.lstrip("0") or "0"
            decimal = "00"

        try:
            # Convert to float and format to two decimal places
            return f"{float(integer + '.' + decimal):.2f}"
        except ValueError:
            return "0.00"

    @staticmethod
    def _convert_amount_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Convert an existing amount column to a standardized decimal format in place.

        Parameters:
        - df: The original DataFrame.
        - column: Name of the column to convert.

        Returns:
        - DataFrame with the column converted.
        """
        if column not in df.columns:
            return df
        df[column] = df[column].apply(BaseProcessor._convert_amount_value)
        return df

    @staticmethod
    def _add_converted_amount_column(
        df: pd.DataFrame, source_column: str, new_column: str
    ) -> pd.DataFrame:
        """
        Create a new column in the DataFrame by converting values from an existing column.

        Parameters:
        - df: The original DataFrame.
        - source_column: Name of the column to convert.
        - new_column: Name of the new column to hold the converted values.

        Returns:
        - DataFrame with the new column added.
        """
        if source_column not in df.columns:
            return df
        df[new_column] = df[source_column].apply(BaseProcessor._convert_amount_value)
        return df


class PostsabuyCODProcessor(BaseProcessor):
    """Processor for Sabuy Speed COD Excel files."""

    def __init__(self, file_path: str, source_config: Dict):
        super().__init__(file_path, source_config.get("date_pattern", ""))
        self.source_config = source_config
        self.date = datetime.utcnow().strftime('%Y%m%d')

    def _convert_buddhist_to_christian(self, year: int) -> int:
        """Converts Buddhist year to Christian year."""
        return year - 543

    def _get_month_number(self, thai_month: str) -> str:
        """Converts Thai month abbreviation to month number."""
        thai_month_map = {
            "ม.ค.": "01", "ก.พ.": "02", "มี.ค.": "03", "เม.ย.": "04",
            "พ.ค.": "05", "มิ.ย.": "06", "ก.ค.": "07", "ส.ค.": "08",
            "ก.ย.": "09", "ต.ค.": "10", "พ.ย.": "11", "ธ.ค.": "12",
            "มกราคม": "01", "กุมภาพันธ์": "02", "มีนาคม": "03", "เมษายน": "04",
            "พฤษภาคม": "05", "มิถุนายน": "06", "กรกฎาคม": "07", "สิงหาคม": "08",
            "กันยายน": "09", "ตุลาคม": "10", "พฤศจิกายน": "11", "ธันวาคม": "12",            
        }
        return thai_month_map.get(thai_month, "")

    def _extract_bill_cycle_from_text(self, header_text):
        """
        Extracts the bill cycle and bill cycle number from the header text.
        This version supports two different date formats in the header.
        """
        bill_cycle_no = None
        bill_cycle = None

        # NORMALIZE WHITESPACE: Replace non-breaking spaces with regular spaces
        ##header_text = header_text.replace('\u00A0', ' ')
        #print(header_text)
        # --- Extract bill_cycle_no (unchanged) ---
        bill_cycle_no_match = re.search(r'ครั้งที่\s*(\d+)', header_text)
        bill_cycle_no = bill_cycle_no_match.group(1) if bill_cycle_no_match else None

        #print(bill_cycle_no)
        # --- Extract bill_cycle (modified) ---

        # Pattern 1: For format "DD Month - DD Month YYYY" (e.g., 31 กรกฎาคม - 4 สิงหาคม 2568)
        # This pattern looks for two separate month names.
        pattern1 = r'ระหว่างวันที่\s*(\d+)\s+([^\s\d-]+)\s*-\s*(\d+)\s+([^\s\d-]+)\s+(\d{4})'
        bill_cycle_match = re.search(pattern1, header_text)
        #print( bill_cycle_match)

        if bill_cycle_match:
            start_day = int(bill_cycle_match.group(1))
            start_thai_month = bill_cycle_match.group(2)
            end_day = int(bill_cycle_match.group(3))
            end_thai_month = bill_cycle_match.group(4)
            buddhist_year = int(bill_cycle_match.group(5))

            start_month = self._get_month_number(start_thai_month)
            end_month = self._get_month_number(end_thai_month)
            year = self._convert_buddhist_to_christian(buddhist_year)
            
            if start_month and end_month:
                # Format: "DD/MM-DD/MM/YYYY"
                bill_cycle = f"{start_day:02d}/{start_month}-{end_day:02d}/{end_month}/{year}"

        # If the first pattern did not match, try the second one.
        if not bill_cycle:
            # Pattern 2: For format "DD - DD Month YYYY" (e.g., 24 - 28 กรกฎาคม 2568)
            # This pattern looks for a single month name after the day range.
            pattern2 = r'ระหว่างวันที่\s*(\d+)\s*-\s*(\d+)\s+([^\s\d-]+)\s+(\d{4})'
            bill_cycle_match = re.search(pattern2, header_text)

            if bill_cycle_match:
                start_day = int(bill_cycle_match.group(1))
                end_day = int(bill_cycle_match.group(2))
                thai_month = bill_cycle_match.group(3)
                buddhist_year = int(bill_cycle_match.group(4))

                month = self._get_month_number(thai_month)
                year = self._convert_buddhist_to_christian(buddhist_year)
                
                if month:
                    # Format: "DD-DD/MM/YYYY"
                    bill_cycle = f"{start_day:02d}-{end_day:02d}/{month}/{year}"

        if not bill_cycle or not bill_cycle_no:
            logger.error(f"Could not extract bill_cycle or bill_cycle_no from header: '{header_text}'")        
    
        return bill_cycle, bill_cycle_no

    def process(self):
        """
        Processes the Sabuy Speed Excel file to transform it into the reconcile format.
        """
        try:
            logger.info(f"Processing file with SabuySpeedProcessor: {self.file_path}")

            # Read the header from cell A1
            header_df = pd.read_excel(self.file_path, header=None, nrows=1, usecols=[0])
            header_text = header_df.iloc[0, 0] if not header_df.empty else ""

            # Extract bill_cycle_no
            bill_cycle, bill_cycle_no = self._extract_bill_cycle_from_text(header_text)

            # Read the main data, assuming headers are on the second row (index 1)
            # and data starts from the third row.
            df = pd.read_excel(self.file_path, skiprows=2, skipfooter=4)
            ###print(df.head(10))

            # Define expected and new column names
            column_mapping = {
                "หมายเลข EMS ": "rccpkey",
                "จำนวนเงิน": "rccamount",
                "วันที่": "source_data_date"
            }
            
            # Select and rename columns
            ##df = df[column_mapping.keys()]
            ##df.rename(columns=column_mapping, inplace=True)
            df['rccpkey'] = df['หมายเลข EMS ']
            df['source_date'] = df['วันที่'].apply(DateConverter.convert_to_yyyymmdd)

            # Add the computed fields
            df["bill_cycle"] = bill_cycle
            df["bill_cycle_no"] = bill_cycle_no
            
            # Ensure correct data types
            self._add_converted_amount_column(df, "จำนวนเงิน", "rccamount")
            ##df['source_date'] = pd.to_datetime(df['source_date'], errors='coerce').dt.strftime('%Y-%m-%d')
            df['rccservice'] = 'POSTSABUY'

            # Reorder columns to the desired reconcile format
            ##final_columns = ['bill_cycle', 'bill_cycle_no', 'rccpkey', 'rccamount', 'source_data_date']
            ##df = df[final_columns]

            # Save the transformed data to a new CSV file
            output_prefix_filename = self.source_config.get("outbound_prefix", None)
            output_filename = f"{output_prefix_filename}_{bill_cycle_no}_{self.date}.csv"
            output_path = os.path.join(os.path.dirname(self.file_path), output_filename)
            self.save_as_csv(df, output_path)

        except Exception as e:
            logger.error(f"Error processing file {self.file_path} with SabuySpeedProcessor: {e}", exc_info=True)
            raise

class FlashProcessor(BaseProcessor):
    """Processor for Pickup Report CSV/Excel files."""

    def __init__(self, file_path: str, source_config: Dict):
        super().__init__(file_path, source_config.get("date_pattern", ""))
        self.source_config = source_config

        # Derive date for output file naming
        name = os.path.basename(file_path).rsplit(".", 1)[0]

        ## check for source flash bulky ###
        self.source_type = 'express'
        if 'bulky' in name:
            self.source_type = 'bulky'

        matches = re.findall(r'\d{8}', name)
        if matches:
            self.date = matches[-1]
        else:
            self.date = datetime.utcnow().strftime('%Y%m%d')
            logger.warning(f"No YYYYMMDD date found in filename {file_path}. Using current date: {self.date}")

    def process(self):
        """
        Processes the Pickup Report file to transform it into the reconcile format.
        """
        try:
            logger.info(f"Processing file with FlashProcessor: {self.file_path}")

            period_value = None
            with open(self.file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if len(lines) > 1 and lines[1].startswith("Period:"):
                    period_value = lines[1].split(":", 1)[1].strip()

            # Detect extension and read file
            if self.file_path.lower().endswith(".csv"):
                df = pd.read_csv(self.file_path, engine = "python", dtype=str, skiprows=2, skipfooter=1)

            # Map required reconcile columns
            df["rccpkey"] = df["Tracking Number"]
            df["rccservice"] = "FLASH-Parcel"
            df["rccamount"] = df["Pickup Total Cost"].apply(self._convert_amount_value)
            df["rccfee"] = "0.00"
            df["source_date"] = df["Pickup Date"].apply(DateConverter.convert_to_yyyymmdd)
            df["batch_date"] = self.date
            df["period"] = period_value  # attach extracted period
            df["source_type_name"] = self.source_type ## add source_type_name to seperate express and bulky

            # Save output file
            output_prefix_filename = self.source_config.get("outbound_prefix", "speed_txn_flash")
            output_filename = f"{output_prefix_filename}_{self.source_type}_{self.date}.csv"
            output_path = os.path.join(os.path.dirname(self.file_path), output_filename)

            self.save_as_csv(df, output_path)
            logger.info(f"Successfully processed and saved {self.file_path} to {output_path}")

        except Exception as e:
            logger.error(f"Error processing file {self.file_path} with PickupReportProcessor: {e}", exc_info=True)
            raise

class FlashCODProcessor(BaseProcessor):
    """Processor for Flash COD Excel files."""

    def __init__(self, file_path: str, source_config: Dict):
        super().__init__(file_path, source_config.get("date_pattern", ""))
        self.source_config = source_config
        # Extract date from filename for the output file
        name = os.path.basename(file_path).rsplit(".", 1)[0]
        # Assuming date is in YYYYMMDD format
        matches = re.findall(r'\d{8}', name)
        if matches:
            self.date = matches[-1]
        else:
            # Fallback to current date if no date found in filename
            self.date = datetime.utcnow().strftime('%Y%m%d')
            logger.warning(f"No YYYYMMDD date found in filename {file_path}. Using current date: {self.date}")

    def process(self):
        """
        Processes the Flash COD Excel file to transform it into the reconcile format.
        """
        try:
            logger.info(f"Processing file with FlashCODProcessor: {self.file_path}")

            sheet_name = self.source_config.get("sheet_name", "COD Detail")
            df = pd.read_excel(self.file_path, sheet_name=sheet_name)

            # Create the new columns based on user's requirements
            df['rccpkey'] = df['Tracking no.']
            df['rccservice'] = 'FLASH'
            ##df['source_date'] = df['Picked up DATE'].apply(DateConverter.convert_to_yyyymmdd)
            ##df['source_date'] = pd.to_datetime(df['Picked up DATE']).dt.strftime('%Y-%m-%d')
            df['source_date'] = df['Picked up DATE'].apply(DateConverter.convert_to_yyyymmdd)
            df['batch_date'] = self.date


            # Use the base method to create a standardized 'rccamount' column
            self._add_converted_amount_column(df, "COD Amount", "rccamount")

            # Select only the required columns for the output file
            #final_columns = ['rccpkey', 'rccservice', 'rccamount', 'source_data_date']
            #df_final = df[final_columns]

            # Save the transformed data to a new CSV file
            output_prefix_filename = self.source_config.get("outbound_prefix", "speed_cod_flash_cod")
            output_filename = f"{output_prefix_filename}_{self.date}.csv"
            output_path = os.path.join(os.path.dirname(self.file_path), output_filename)
            self.save_as_csv(df, output_path)
            logger.info(f"Successfully processed and saved {self.file_path} to {output_path}")

        except Exception as e:
            logger.error(f"Error processing file {self.file_path} with FlashCODProcessor: {e}", exc_info=True)
            raise    

class SpeedTransactionProcessor(BaseProcessor):
    """Processor for SPD CSV files -> reconcile format.
    Pattern aligned with FlashProcessor.process()  :contentReference[oaicite:0]{index=0}
    FLASH and POSTSABUY 
    """

    def __init__(self, file_path: str, source_config: Dict):
        super().__init__(file_path, source_config.get("date_pattern", ""))
        self.source_config = source_config
        # Derive date for output naming from filename (YYYYMMDD) or fallback to UTC today
        name = os.path.basename(file_path).rsplit(".", 1)[0]
        matches = re.findall(r"\d{8}", name)
        if matches:
            self.date = matches[-1]
        else:
            self.date = datetime.utcnow().strftime("%Y%m%d")
            logger.warning(f"No YYYYMMDD date found in filename {file_path}. Using current date: {self.date}")

    def process(self):
        """
        Transform SPD ShipSmile CSV into reconcile file format:
          - rccpkey   = tracking_code
          - rccamount = cost_price
          - rccfee    = "0.00"
          - source_date = trns_date (converted to YYYYMMDD)
          - rccservice = map sub_product_name -> 'FLASH' for FLSESP/FLEB, else 'POSTSABUY'
        Keeps all original columns and appends the reconcile columns.
        """
        try:
            logger.info(f"Processing file with SpeedTransactionProcessor: {self.file_path}")

            # Read CSV as strings
            df = pd.read_csv(self.file_path, dtype=str)

            # Core reconcile fields
            df["rccpkey"] = df["tracking_code"]
            # Map to known courier; default to POSTSABUY if not Flash family
            df["rccservice"] = np.where(
                df["sub_product_name"].isin(["FLSESP", "FLE","FLEB","FLEF","FLES","FLEB", "FLASH", "FLASHBULKY"]),
                "FLASH-Parcel",
                "POSTSABUY-Parcel",
            )

            df["rccamount"] = df["cost_price"].apply(self._convert_amount_value)
            df["rccfee"] = "0.00"
            df["source_date"] = df["trns_date"].apply(DateConverter.convert_to_yyyymmdd)
            df["batch_date"] = self.date

            # Save output
            output_prefix = self.source_config.get("outbound_prefix", "speed_txn")
            output_filename = f"{output_prefix}_{self.date}.csv"
            output_path = os.path.join(os.path.dirname(self.file_path), output_filename)
            self.save_as_csv(df, output_path)
            logger.info(f"Successfully processed and saved {self.file_path} to {output_path}")

        except Exception as e:
            logger.error(f"Error processing file {self.file_path} with SpeedTransactionProcessor: {e}", exc_info=True)
            raise


class BankStatementProcessor(BaseProcessor):
    """Processes a multi-sheet Excel bank statement file."""

    def __init__(self, file_path: str, source_config: Dict):
        """
        Initializes the processor for bank statements.
        Extracts the date from the filename (e.g., '25062025.xlsx') and reformats it.
        """
        ##super().__init__(file_path, source_config.get("date_pattern_in_filename", ""))
        self.source_config = source_config
        self.file_path = file_path
        self.path_name = os.path.dirname(file_path)
        
        # Extract date like '25062025' from filename
        ##filename_date_str = self._get_date_from_filename()
        ##if not filename_date_str:
            ###raise ValueError(f"Could not extract date from filename: {os.path.basename(file_path)}")
            # Fallback to current date if no date found in filename
        self.output_date = datetime.utcnow().strftime('%Y%m%d')
        logger.warning(f"No YYYYMMDD date found in filename {file_path}. Using current date: {self.output_date}")            
        
        # Convert DDMMYYYY to YYYYMMDD for the output file name
        ##self.output_date = DateConverter.convert_to_yyyymmdd(filename_date_str)
        logger.info(f"Initialized BankStatementProcessor for {file_path}. Output date: {self.output_date}")

    def process(self):
        """
        Main processing function. Opens the Excel file and processes each configured sheet.
        """
        try:
            xls = pd.ExcelFile(self.file_path)
            for sheet_config in self.source_config.get("sheets", []):
                if sheet_config['name'] in xls.sheet_names:
                    logger.info(f"Processing sheet: {sheet_config['name']}")
                    self._process_sheet(xls, sheet_config)
                else:
                    logger.warning(f"Sheet '{sheet_config['name']}' not found in {self.file_path}. Skipping.")
        except Exception as e:
            logger.error(f"Failed to process file {self.file_path}: {e}", exc_info=True)
            raise

    def _process_sheet(self, xls: pd.ExcelFile, config: Dict):
        """Processes a single sheet based on its configuration."""
        try:
            # 1. Read the specific sheet with skip rows/footers
            df = pd.read_excel(
                xls,
                sheet_name=config['name'],
                skiprows=config.get('skip_header', 0),
                skipfooter=config.get('skip_footer', 0),
                dtype=str  # Read all as string to avoid type issues
            )
            df.dropna(how='all', inplace=True) # Drop rows that are completely empty

            # 2. Apply filters
            if 'filters' in config:
                for f in config['filters']:
                    col = f['column']
                    val = f['value']
                    if col not in df.columns:
                        logger.warning(f"Filter column '{col}' not found in sheet '{config['name']}'. Skipping filter.")
                        continue
                    
                    df[col] = df[col].astype(str).str.strip()
                    if f['match_type'] == 'exact':
                        df = df[df[col] == str(val)]
                    elif f['match_type'] == 'isin':
                        df = df[df[col].isin([str(v) for v in val])]
                    elif f['match_type'] == 'contains':
                        df = df[df[col].str.contains(str(val), na=False)]

            if df.empty:
                logger.info(f"Sheet '{config['name']}' is empty after filtering. No output will be generated.")
                return

            # 3. Handle aggregation (for SCB)
            if config.get('aggregation'):
                agg_config = config['aggregation']
                group_by_cols = agg_config['group_by']
                agg_col = agg_config['agg_column']
                
                # Convert amount column to numeric before aggregation
                df[agg_col] = pd.to_numeric(df[agg_col].apply(self._convert_amount_value), errors='coerce')

                df = df.groupby(group_by_cols).agg(
                    rccamount=(agg_col, agg_config['agg_func'])
                ).reset_index()
                
                # After aggregation, format amount back to string
                df['rccamount'] = df['rccamount'].apply(lambda x: f"{x:.2f}")

            else:
                # 4. Standard amount conversion for non-aggregated sheets
                self._add_converted_amount_column(df, config['mapping']['rccamount'], 'rccamount')

            # 5. Map columns and create standard fields
            mapping = config['mapping']
            df['source_date'] = df[mapping['source_date']].apply(DateConverter.convert_to_yyyymmdd)
            df['rccservice'] = config['rccservice']

            # 6. Create rccpkey
            pkey_template = config.get('rccpkey_template')
            if pkey_template:
                # Format based on template, e.g., "AGCWTR|{เงินออกบัญชี}"
                # This uses f-string like capabilities on dataframe columns
                df['rccpkey'] = df.apply(lambda row: pkey_template.format(**row), axis=1)
            else:
                # Simple mapping from a single column
                df['rccpkey'] = df[mapping['rccpkey']]

            # 7. Select final columns and save
            ##final_columns = ['rccpkey', 'rccservice', 'rccamount', 'source_date']
            ##df_final = df[final_columns]
            
            # Generate output filename
            sheet_name_cleaned = re.sub(r'[^a-zA-Z0-9_-]', '', config['name'])
            output_filename = f"bank_agent_{sheet_name_cleaned}_{self.output_date}.csv"
            output_path = os.path.join(self.path_name, output_filename)
            
            ##self._save_csv(df_final, output_path)
            self._save_csv(df, output_path)


        except Exception as e:
            logger.error(f"Failed to process sheet '{config['name']}': {e}", exc_info=True)

class XLSBProcessor(BaseProcessor):
    """Processes XLSB files and generates CSV outputs."""

    def __init__(self, file_path: str, sheet_name: str, date_pattern: str):
        """Initialize XLSB processor with file path, sheet name, and date pattern."""
        self.file_path = file_path
        self.path_name = os.path.dirname(file_path)
        self.sheet_name = sheet_name
        ##self.date = DateTransformer.extract_from_filename(file_path, date_pattern)
        name = file_path.rsplit(".", 1)[0]

        # Find all matches for YYYYMMDD (8 digits) or DD.MM.YY (2.2.2)
        matches = re.findall(r"\d{8}|\d{2}\.\d{2}\.\d{2}", name)

        # If no matches found, return None (could raise an error if preferred)
        if not matches:
            raise ValueError(
                f"File {self.file_path} date {matches} in filename does not matched with date format"
            )

        # Take the last match, as the date is typically near the end
        date_str = matches[-1]


        ##self.date = DateTransformer.extract_date(file_path)
        self.date = DateConverter.convert_to_yyyymmdd(date_str)
        logger.info(f"Initialized XLSB processor for {file_path} - {sheet_name}")

    def _read_xlsb(self) -> pd.DataFrame:
        """Read XLSB file with standard settings."""
        kwargs = {"engine": "pyxlsb", "dtype": "str"}
        if self.sheet_name.startswith("BBL"):
            kwargs.update({"skiprows": 6, "skipfooter": 14})
        return pd.read_excel(self.file_path, sheet_name=self.sheet_name, **kwargs)

    def _read_xlsb_report(self) -> pd.DataFrame:
        """Read XLSB report file, identifying header dynamically."""
        df = pd.read_excel(
            self.file_path,
            sheet_name=self.sheet_name,
            engine="pyxlsb",
            dtype=str,
            header=None,
        )
        header_idx = next((idx for idx, row in df.iterrows() if row[0] == "No."), None)
        if header_idx is None:
            raise ValueError("Header row not found")
        df.columns = df.iloc[header_idx]
        df_data = df.iloc[header_idx + 1 :]
        df_data = df_data[pd.to_numeric(df_data["No."], errors="coerce").notna()]
        return df_data.reset_index(drop=True)

    def _process_bbl2(self, df: pd.DataFrame) -> None:
        """Process BBL-specific sheet data."""
        df["AMOUNT"] = df["AMOUNT"].astype(str).str.replace(",", "")
        df["REFERENCE NO."] = df["REFERENCE NO."].astype(str).str.strip()
        ## Modify 19-05-2025 by nwt: update to DateConverter
        ##df["PAY.DATE"] = df["PAY.DATE"].apply(DateTransformer.thai_to_yyyymmdd)
        df["PAY.DATE"] = df["PAY.DATE"].apply(DateConverter.convert_to_yyyymmdd)
        filters = {
            "gameonline": ("SBY", "SQ"),
            "jsk_vending": lambda x: not x.startswith(("SBY", "SQ", "ACMW")),
        }

        def get_suffix(ref):
            for prefix, condition in filters.items():
                if callable(condition) and condition(ref):
                    return prefix
                elif isinstance(condition, tuple) and ref.startswith(condition):
                    return prefix
            return "unknown"

        df["rccservice"] = df["REFERENCE NO."].apply(get_suffix)
        df["rccpkey"] = (
            df["CUSTOMER NO."].astype(str)
            + "|"
            + df["REFERENCE NO."].astype(str)
            + "|"
            + df["rccservice"]
        )
        df["source_date"] = df["PAY.DATE"]
        df_filtered = df[df["rccservice"] != "unknown"]

        self._add_converted_amount_column(df_filtered, "AMOUNT", "rccamount")

        self._save_csv(
            df_filtered,
            os.path.join(self.path_name, f"merchant_epay_BBL_{self.date}.csv"),
        )

    def _process_scb2(self, df: pd.DataFrame) -> None:
        """Process SCB-specific sheet data."""
        df["Ref. 2"] = df["Ref. 2"].astype(str).str.strip()
        df["Amount"] = df["Amount"].astype(str).str.replace(",", "")
        ##df["Date"] = df["Date"].apply(DateTransformer.thai_to_yyyymmdd)
        ## apply new DateCoverter class
        df["Date"] = df["Date"].apply(DateConverter.convert_to_yyyymmdd)
        ## temporary add bpos filter for 00002 case
        filters = {
            "bpos": ("PB", "T2", "00002"),
            "pos": "17",
            "easy": "M",
            "thaivan": (
                "BANGKAPI",
                "SEACON",
                "SIAM",
                "UNION",
                "CENTRAL",
                "THAMASAT",
                "SAMUTPRA",
                "THAIVAN"
            ),
        }

        def get_suffix(ref):
            for prefix, value in filters.items():
                if isinstance(value, str) and ref.startswith(value):
                    return prefix
                elif isinstance(value, tuple) and any(ref.startswith(v) for v in value):
                    return prefix
            return "unknown"

        df["rccservice"] = df["Ref. 2"].apply(get_suffix)

        ##df['rccpkey'] = df['Ref. 1'].astype(str) + '|' + df['Ref. 2'].astype(str) + '|' + df['rccservice']

        df["rccpkey"] = np.where(
            df["rccservice"].isin(["pos"]),
            df["Ref. 1"].astype(str) + "|" + df["rccservice"],
            df["Ref. 1"].astype(str)
            + "|"
            + df["Ref. 2"].astype(str)
            + "|"
            + df["rccservice"],
        )

        df["source_date"] = df["Date"]
        df_filtered = df[df["rccservice"] != "unknown"]

        self._add_converted_amount_column(df_filtered, "Amount", "rccamount")

        self._save_csv(
            df_filtered,
            os.path.join(self.path_name, f"merchant_epay_SCB_{self.date}.csv"),
        )

    def process(self, processor_method: str) -> None:
        """Execute processing based on the specified method."""
        try:
            df = (
                self._read_xlsb_report()
                if processor_method == "bbl2"
                else self._read_xlsb()
            )
            getattr(self, f"_process_{processor_method}")(df)
            logger.info(f"Completed processing {self.sheet_name}")
        except Exception as e:
            logger.error(f"Processing failed for {self.sheet_name}: {e}")
            raise

class KBankQR22Processor(BaseProcessor):
    """Processes KBank QR CSV files for Merchant E Payment."""

    def __init__(self, file_path: str, source_config: Dict):
        super().__init__(file_path, source_config.get("date_pattern", ""))
        self.source_config = source_config
        self.output_prefix = self.source_config.get("outbound_prefix","merchant_epay_kbankqr")

        name = os.path.basename(file_path)
        match = re.search(r'(\d{8})', name)
        if match:
            self.date = match.group(1)
        else:
            self.date = datetime.utcnow().strftime('%Y%m%d')
            logger.warning(f"No YYYYMMDD date found in filename {file_path}. Using current date: {self.date}")

    def process(self):
        """
        Processes the KBankQR22 CSV file to transform it into the reconcile format,
        keeping all original columns and adding new ones.
        """
        try:
            logger.info(f"Processing file with KBankQR22rocessor: {self.file_path}")
         
            df = pd.read_csv(self.file_path, dtype=str)

            filters = {
                "gameonline": ("SBY", "SQ"),
                "bpos": ("PB", "T2", "00002"),
                "pos": ("17",),
                "easy": ("M",),
                "thaivan": (
                    "BANGKAPI",
                    "SEACON",
                    "SIAM",
                    "UNION",
                    "CENTRAL",
                    "THAMASAT",
                    "SAMUTPRA",
                    "THAIVAN"
                ),                
                "jsk_vending": lambda x: not x.startswith(("SBY", "SQ", "ACMW", "30010002")),
            }

            def get_suffix(ref):
                for prefix, condition in filters.items():
                    if callable(condition) and condition(ref):
                        return prefix
                    elif isinstance(condition, tuple) and ref.startswith(condition):
                        return prefix
                return "unknown"        

            df["rccservice"] = df["REF2"].apply(get_suffix)

            # Add new columns based on the provided mapping
            ##df['rccpkey'] = df["REF1"] + "|" + df["REF2"] + "|" + df['rccservice']

            df["rccpkey"] = np.where(
                df["rccservice"].isin(["pos"]),
                df["REF1"].astype(str) + "|" + df["rccservice"],
                df["REF1"].astype(str)
                + "|"
                + df["REF2"].astype(str)
                + "|"
                + df["rccservice"],
            )            
            
            df['source_date'] = df['OriginalTransactionDate'].apply(DateConverter.convert_to_yyyymmdd)
            self._add_converted_amount_column(df, "Amount", "rccamount")
            df['rccfee'] = "0.00"

            df_filtered = df[df["rccservice"] != "unknown"]
            # Save the transformed data (including all original columns) to a new CSV file
            output_filename = f"{self.output_prefix}_{self.date}.csv"
            output_path = os.path.join(os.path.dirname(self.file_path), output_filename)
            self.save_as_csv(df_filtered, output_path)

        except Exception as e:
            logger.error(f"Error processing file {self.file_path} with KBankQR22rocessor: {e}", exc_info=True)
            raise

class BankAgentTxnProcessor(BaseProcessor):
    """Processes bank agent transaction CSV files."""

    def __init__(self, file_path: str, source_config: Dict):
        super().__init__(file_path, source_config.get("date_pattern", ""))
        self.source_config = source_config
        self.output_prefix = self.source_config.get("outbound_prefix","local_bankagent")

        name = os.path.basename(file_path)
        match = re.search(r'(\d{8})', name)
        if match:
            self.date = match.group(1)
        else:
            self.date = datetime.utcnow().strftime('%Y%m%d')
            logger.warning(f"No YYYYMMDD date found in filename {file_path}. Using current date: {self.date}")

    def process(self):
        """
        Processes the bank agent transaction CSV file to transform it into the reconcile format.
        """
        try:
            logger.info(f"Processing file with BankAgentTxnProcessor: {self.file_path}")

            df = pd.read_csv(self.file_path, dtype=str)

            #### filter out GSBDEPOSITONLINE because no need for current reconcile
            df = df[df['order_service'] != 'GSBDEPOSITONLINE']

            key_map = {
                'KTBDEPOSIT': 'order_ref7',
                'BAACDEPOSIT': 'batchdate',
                'BAACDEPOSITPOS': 'batchdate',
                'CIMBDEPOSIT': 'order_gwref',
                'BAYDEPOSIT': 'order_created',
                'SCBDEPOSIT': 'batchdate',
                'SCBDEPOSITPOS': 'batchdate',
                'BBLDEPOSIT': 'batchdate',
                'KBANKDEPOSIT': 'order_ref',
                'GSBDEPOSITONLINE': 'order_ref'
            }

            ##df['rccpkey'] = df.apply(lambda row: row[key_map.get(row['order_service'], 'order_ref')], axis=1)
            # 1. Define a function to get and format the key
            def get_formatted_key(row):
                """
                Determines the source column and formats the value
                if the source is 'batchdate'.
                """
                source_column = key_map.get(row['order_service'], 'order_ref')
                value = row[source_column]

                if row['order_service'] == 'BAYDEPOSIT':
                    # Format order_created as YYYYMMDDHHMM
                    return pd.to_datetime(value).strftime('%Y%m%d%H%M')
                elif source_column == 'batchdate':
                    # Convert to datetime and format as YYYYMMDD
                    return pd.to_datetime(value).strftime('%Y%m%d')
                else:
                    # Otherwise, return the original value
                    return value

            # 2. Apply the function to create the 'rccpkey' column
            df['rccpkey'] = df.apply(get_formatted_key, axis=1)

            df['rccservice'] = np.where(df['order_service'] == 'BAACDEPOSITPOS', 'BAACDEPOSIT', df['order_service'])
            df['rccamount'] = df['order_amount'].apply(self._convert_amount_value)
            fee_map = {
                'BAYDEPOSIT': '10.00',
                'KTBDEPOSIT': '10.00',
                'SCBDEPOSIT': '10.00',
                'SCBDEPOSITPOS': '15.00',
                'BAACDEPOSIT': '5.00',
                'BAACDEPOSITPOS': '5.00',
                'CIMBDEPOSIT': '0.00', ## external CIMB does not have fee amount.
                'KBANKDEPOSIT': '10.00'
            }
            # 1. Apply the map for all standard cases. This will result in NaN for 'BBL'.
            df['rccfee'] = df['order_service'].map(fee_map)

            # 2. Use .loc to find rows where order_service is 'BBL' and assign the value from 'order_fee'.
            #    To avoid mixed data types, it's good practice to cast order_fee to string.
            is_bbl = df['order_service'] == 'BBLDEPOSIT'
            df.loc[is_bbl, 'rccfee'] = df.loc[is_bbl, 'order_fee'].astype(str)

            # 3. Fill any remaining missing values (for services not in the map and not 'BBL') with '0.00'.
            df['rccfee'] = df['rccfee'].fillna('0.00')

            ##df['source_date'] = pd.to_datetime(df['batchdate']).dt.strftime('%Y-%m-%d')
            df['source_date'] = df['batchdate'].apply(DateConverter.convert_to_yyyymmdd)

            output_filename = f"{self.output_prefix}_{self.date}.csv"
            output_path = os.path.join(os.path.dirname(self.file_path), output_filename)
            self.save_as_csv(df, output_path)

        except Exception as e:
            logger.error(f"Error processing file {self.file_path} with BankAgentTxnProcessor: {e}", exc_info=True)
            raise


class KBankStatementProcessor(BaseProcessor):
    """Processes KBank statement CSV files."""

    def __init__(self, file_path: str, source_config: Dict):
        super().__init__(file_path, source_config.get("date_pattern", ""))
        self.source_config = source_config
        self.output_prefix = self.source_config.get("outbound_prefix","bank_kbank_stm")

        name = os.path.basename(file_path)
        match = re.search(r'(\d{8})', name)
        if match:
            self.date = match.group(1)
        else:
            self.date = datetime.utcnow().strftime('%Y%m%d')
            logger.warning(f"No YYYYMMDD date found in filename {file_path}. Using current date: {self.date}")

    def process(self):
        """
        Processes the KBank statement CSV file to transform it into the reconcile format,
        keeping all original columns and adding new ones.
        """
        try:
            logger.info(f"Processing file with KBankStatementProcessor: {self.file_path}")

            # Read the CSV file, skipping the initial metadata rows (first 8 rows).
            # The actual header is on row 9.
            df = pd.read_csv(self.file_path, skiprows=8, skipfooter=6, dtype=str)

            # Add new columns based on the provided mapping
            df['rccpkey'] = df['Merchant Transaction ID']
            df['rccservice'] = 'KBANKDEPOSIT'
            df['source_date'] = df['Date Time'].apply(DateConverter.convert_to_yyyymmdd)
            self._add_converted_amount_column(df, "Amount", "rccamount")
            self._add_converted_amount_column(df, "Fee", "rccfee")

            # Save the transformed data (including all original columns) to a new CSV file
            output_filename = f"{self.output_prefix}_{self.date}.csv"
            output_path = os.path.join(os.path.dirname(self.file_path), output_filename)
            self.save_as_csv(df, output_path)

        except Exception as e:
            logger.error(f"Error processing file {self.file_path} with KBankStatementProcessor: {e}", exc_info=True)
            raise


class BAACStatementProcessor(BaseProcessor):
    """Processes BAAC statement CSV files."""

    def __init__(self, file_path: str, source_config: Dict):
        super().__init__(file_path, source_config.get("date_pattern", ""))
        self.source_config = source_config
        self.output_prefix = self.source_config.get("outbound_prefix","bank_baac_stm")

        name = os.path.basename(file_path)
        # Try to find the newer format: stm_ktb_..._YYYYMMDDHHMMSS.xls
        match_new = re.search(r'_(\d{14})\.', name)
        if match_new:
            timestamp_str = match_new.group(1)
            date_part_str = timestamp_str[:8]
            try:
                # The report is for the previous day
                report_date = datetime.strptime(date_part_str, '%Y%m%d')
                self.date = report_date.strftime('%Y%m%d')
                logger.info(f"Extracted date {self.date} from new KTB filename format: {name}")
            except ValueError:
                logger.warning(f"Could not parse date from {name}, falling back to default.")
                self.date = datetime.utcnow().strftime('%Y%m%d')
        else:
            # Fallback to the original logic for older formats
            match_old = re.search(r'(\d{8})', name)
            if match_old:
                self.date = match_old.group(1)
            else:
                self.date = datetime.utcnow().strftime('%Y%m%d')
                logger.warning(f"No YYYYMMDD date found in filename {file_path}. Using current date: {self.date}")

        self.header_columns = [
            "a", "code", "account_no", "date", "bank_tran_id", "account_name",
            "branch_code", "branch_name", "address1", "address2", "date1",
            "trans_code", "resv1", "deposit_amount", "trans_type",
            "withdraw_amount", "balance", "resv2", "bank_branch", "remark"
        ]

    def process(self):
        """
        Processes the BAAC statement CSV file.
        """
        try:
            logger.info(f"Processing file with BAACStatementProcessor: {self.file_path}")

            df = pd.read_csv(
                self.file_path,
                header=None,
                delimiter='|',
                names=self.header_columns,
                dtype=str,
                encoding='TIS-620'
            )

            df = df[df['trans_code'].str.strip().isin(['AGCWTR', 'SPCD10'])]

            df['rccpkey'] = self.date  ##use batch_date for many-to-one case
            df['rccservice'] = 'BAACDEPOSIT'
            df['source_date'] = df['date'].apply(DateConverter.thai_yymmdd_to_yyyymmdd)
            self._add_converted_amount_column(df, "deposit_amount", "rccamount")

            output_filename = f"{self.output_prefix}_{self.date}.csv"
            output_path = os.path.join(os.path.dirname(self.file_path), output_filename)
            self.save_as_csv(df, output_path)

        except Exception as e:
            logger.error(f"Error processing file {self.file_path} with BAACStatementProcessor: {e}", exc_info=True)
            raise


class CIMBStatementProcessor(BaseProcessor):
    """Processes CIMB statement Excel files from a ZIP archive."""

    def __init__(self, file_path: str, source_config: Dict):
        super().__init__(file_path, source_config.get("date_pattern", ""))
        self.source_config = source_config
        self.output_prefix = self.source_config.get("outbound_prefix","bank_cimb_stm")

        name = os.path.basename(file_path)
        match = re.search(r'(\d{8})', name)
        if match:
            self.date = match.group(1)
        else:
            self.date = datetime.utcnow().strftime('%Y%m%d')
            logger.warning(f"No YYYYMMDD date found in filename {file_path}. Using current date: {self.date}")
 
    def process(self):
        """
        Processes the CIMB statement Excel file.
        """
        try:
            logger.info(f"Processing file with CIMBStatementProcessor: {self.file_path}")

            # The file path here is the extracted .xlsx file
            
            df = pd.read_excel(self.file_path)

            ## filter only "Status": "Completed"
            df = df[df['Status'].str.strip() == 'Completed']

            if df.empty:
                logger.info(f"File {self.file_path} is empty or contains only headers. Skipping processing.")
                return

            # Check for "NO DATA IN REPORT" case in the first data cell
            if "NO DATA IN REPORT" in str(df.iloc[0, 0]):
                logger.info(f"File {self.file_path} contains 'NO DATA IN REPORT'. Skipping processing.")
                return

            # Map columns based on requirements
            df['rccpkey'] = df['Client Transaction No']
            df['rccservice'] = 'CIMBDEPOSIT'
            df['source_date'] = df['ValueDateTime'].apply(DateConverter.convert_to_yyyymmdd)
            self._add_converted_amount_column(df, "Amount", "rccamount")

            # Define output file
            output_filename = f"{self.output_prefix}_{self.date}.csv"
            output_path = os.path.join(os.path.dirname(self.file_path), output_filename)
            self.save_as_csv(df, output_path)

        except Exception as e:
            logger.error(f"Error processing file {self.file_path} with CIMBStatementProcessor: {e}", exc_info=True)
            raise


class KTBStatementProcessor(BaseProcessor):
    """Processes KTB statement Excel files."""

    def __init__(self, file_path: str, source_config: Dict):
        super().__init__(file_path, source_config.get("date_pattern", ""))
        self.source_config = source_config
        self.output_prefix = self.source_config.get("outbound_prefix","bank_ktb_stm")

        name = os.path.basename(file_path)
        
        # Try to find the newer format: stm_ktb_..._YYYYMMDDHHMMSS.xls
        match_new = re.search(r'_(\d{14})\.', name)
        if match_new:
            timestamp_str = match_new.group(1)
            date_part_str = timestamp_str[:8]
            try:
                # The report is for the previous day
                report_date = datetime.strptime(date_part_str, '%Y%m%d')
                self.date = report_date.strftime('%Y%m%d')
                logger.info(f"Extracted date {self.date} from new KTB filename format: {name}")
            except ValueError:
                logger.warning(f"Could not parse date from {name}, falling back to default.")
                self.date = datetime.utcnow().strftime('%Y%m%d')
        else:
            # Fallback to the original logic for older formats
            match_old = re.search(r'(\d{8})', name)
            if match_old:
                self.date = match_old.group(1)
            else:
                self.date = datetime.utcnow().strftime('%Y%m%d')
                logger.warning(f"No YYYYMMDD date found in filename {file_path}. Using current date: {self.date}")

    def process(self):
        """
        Processes the KTB statement Excel file by pairing transaction and fee rows.
        """
        try:
            logger.info(f"Processing file with KTBStatementProcessor: {self.file_path}")

            df = pd.read_excel(self.file_path, skiprows=10)
            df.columns = df.columns.str.strip()  # Clean column names

            # The user specified that the description column is next to 'Transaction Code'
            # and may not have a header. We'll identify it by position.
            try:
                trans_code_col_index = df.columns.get_loc('Transaction Code')
                description_col_name = df.columns[trans_code_col_index + 1]
            except KeyError:
                logger.error("'Transaction Code' column not found. Cannot identify description column.")
                raise

            processed_data = []

            for i, row in df.iterrows():
                if str(row['Transaction Code']).strip() == 'BADWT':
                    # This is a main transaction row, extract its data
                    try:
                        # The rccpkey comes from the description column
                        description = str(row[description_col_name])
                        rccpkey = description.split('/')[1]
                    except IndexError:
                        logger.warning(f"Could not extract rccpkey from '{description}' in row {i}. Skipping.")
                        continue
                    
                    # Convert the row to a dictionary to preserve original columns
                    transaction_data = row.to_dict()

                    # Add/overwrite transformed columns
                    transaction_data['rccpkey'] = rccpkey
                    transaction_data['rccservice'] = 'KTBDEPOSIT'
                    transaction_data['source_date'] = DateConverter.convert_to_yyyymmdd(row['Date'])
                    transaction_data['rccamount'] = self._convert_amount_value(row['Amount'])
                    
                    # The fee is in the next row
                    rccfee = "0.00"
                    if (i + 1) < len(df) and str(df.iloc[i + 1]['Transaction Code']).strip() == 'BADFE':
                        rccfee = self._convert_amount_value(df.iloc[i + 1]['Amount'])
                    transaction_data['rccfee'] = rccfee

                    processed_data.append(transaction_data)

            if not processed_data:
                logger.info(f"No valid KTB transaction data found in {self.file_path}")
                return

            processed_df = pd.DataFrame(processed_data)

            output_filename = f"{self.output_prefix}_{self.date}.csv"
            output_path = os.path.join(os.path.dirname(self.file_path), output_filename)
            self.save_as_csv(processed_df, output_path)

        except Exception as e:
            logger.error(f"Error processing file {self.file_path} with KTBStatementProcessor: {e}", exc_info=True)
            raise


class SCBStatementProcessor(BaseProcessor):
    """Processes SCB statement CSV files."""

    def __init__(self, file_path: str, source_config: Dict):
        super().__init__(file_path, source_config.get("date_pattern", ""))
        self.source_config = source_config
        self.output_prefix = self.source_config.get("outbound_prefix","bank_scb_stm")
        
        name = os.path.basename(file_path)
        # Try to find the newer format: stm_ktb_..._YYYYMMDDHHMMSS.xls
        match_new = re.search(r'_(\d{14})\.', name)
        if match_new:
            timestamp_str = match_new.group(1)
            date_part_str = timestamp_str[:8]
            try:
                # The report is for the previous day
                report_date = datetime.strptime(date_part_str, '%Y%m%d')
                self.date = report_date.strftime('%Y%m%d')
                logger.info(f"Extracted date {self.date} from new SCB filename format: {name}")
            except ValueError:
                logger.warning(f"Could not parse date from {name}, falling back to default.")
                self.date = datetime.utcnow().strftime('%Y%m%d')
        else:
            # Fallback to the original logic for older formats
            match_old = re.search(r'(\d{8})', name)
            if match_old:
                self.date = match_old.group(1)
            else:
                self.date = datetime.utcnow().strftime('%Y%m%d')
                logger.warning(f"No YYYYMMDD date found in filename {file_path}. Using current date: {self.date}")

    def process(self):
        """
        Processes the SCB statement CSV file by pairing transaction and fee rows.
        """
        try:
            logger.info(f"Processing file with SCBStatementProcessor: {self.file_path}")

            df = pd.read_csv(self.file_path)
            df.columns = df.columns.str.strip()

            processed_data = []

            for i, row in df.iterrows():
                if str(row['Transaction Code']).strip() == 'CW':
                    transaction_data = row.to_dict()

                    ##transaction_data['rccpkey'] = self._convert_amount_value(row['Debit Amount'])
                    transaction_data['rccpkey'] = DateConverter.convert_to_yyyymmdd(row['Date'])

                    ##transaction_data['rccservice'] = 'SCBDEPOSIT'
                    transaction_data['rccservice'] = np.where(transaction_data['Account Number'] == 1323024925, 'SCBDEPOSITPOS', 'SCBDEPOSIT')
                    transaction_data['rccamount'] = self._convert_amount_value(row['Debit Amount'])
                    transaction_data['source_date'] = DateConverter.convert_to_yyyymmdd(row['Date'])

                    rccfee = "0.00"
                    if (i + 1) < len(df) and str(df.iloc[i + 1]['Transaction Code']).strip() == 'FE':
                        rccfee = self._convert_amount_value(df.iloc[i + 1]['Debit Amount'])
                    transaction_data['rccfee'] = rccfee

                    processed_data.append(transaction_data)

                elif str(row['Transaction Code']).strip() == 'XW':
                    desc = str(row['Description']).lower()

                    # Only process when this is the "deposit" line
                    if "deposit" in desc:
                        transaction_data = row.to_dict()
                        transaction_data['rccpkey'] = DateConverter.convert_to_yyyymmdd(row['Date'])
                        transaction_data['rccservice'] = np.where(
                            transaction_data['Account Number'] == 1323024925,
                            'SCBDEPOSITPOS',
                            'SCBDEPOSIT'
                        )
                        transaction_data['source_date'] = DateConverter.convert_to_yyyymmdd(row['Date'])

                        # rccamount comes from this deposit row
                        transaction_data['rccamount'] = self._convert_amount_value(row['Debit Amount'])
                        transaction_data['rccfee'] = np.where(
                            transaction_data['rccservice'] == 'SCBDEPOSITPOS', "15.00", "10.00"
                        )

                        processed_data.append(transaction_data)

                    # if it's a "fee" line → skip it (because already consumed)
                    elif "fee" in desc:
                        continue                  

            if not processed_data:
                logger.info(f"No valid SCB transaction data found in {self.file_path}")
                return

            processed_df = pd.DataFrame(processed_data)

            ##output_filename = f"bank_scb_stm_{self.date}.csv"
            output_filename = f"{self.output_prefix}_{self.date}.csv"
            output_path = os.path.join(os.path.dirname(self.file_path), output_filename)
            self.save_as_csv(processed_df, output_path)

        except Exception as e:
            logger.error(f"Error processing file {self.file_path} with SCBStatementProcessor: {e}", exc_info=True)
            raise

class BAYStatementProcessor(BaseProcessor):
    """Processes BAY statement Excel files."""

    def __init__(self, file_path: str, source_config: Dict):
        super().__init__(file_path, source_config.get("date_pattern", ""))
        self.source_config = source_config
        self.output_prefix = self.source_config.get("outbound_prefix","bank_bay_stm")

        name = os.path.basename(file_path)
        # Try to find the newer format: stm_ktb_..._YYYYMMDDHHMMSS.xls
        match_new = re.search(r'_(\d{14})\.', name)
        if match_new:
            timestamp_str = match_new.group(1)
            date_part_str = timestamp_str[:8]
            try:
                # The report is for the previous day
                report_date = datetime.strptime(date_part_str, '%Y%m%d')
                self.date = report_date.strftime('%Y%m%d')
                logger.info(f"Extracted date {self.date} from new KTB filename format: {name}")
            except ValueError:
                logger.warning(f"Could not parse date from {name}, falling back to default.")
                self.date = datetime.utcnow().strftime('%Y%m%d')
        else:
            # Fallback to the original logic for older formats
            match_old = re.search(r'(\d{8})', name)
            if match_old:
                self.date = match_old.group(1)
            else:
                self.date = datetime.utcnow().strftime('%Y%m%d')
                logger.warning(f"No YYYYMMDD date found in filename {file_path}. Using current date: {self.date}")

    def process(self):
        """
        Processes the BAY statement Excel file by linking transaction (TW) and fee (CM) rows.
        """
        try:
            logger.info(f"Processing file with BAYStatementProcessor: {self.file_path}")

            df = pd.read_excel(self.file_path, dtype={'TXN_REF': str})

            df.columns = df.columns.str.strip()

            # Ensure TIME column is a string for consistent key creation
            df['TIME'] = df['TIME'].astype(str)

            # Create a mapping from a composite key (TXN_REF) to the fee amount for all 'CM' rows
            #fee_map = df[df['TXNCODE'].str.strip() == 'CM'].set_index(['TXN_REF'])['DEBIT']
            # Extract CM (fee) rows
            fee_df = df[df['TXNCODE'].str.strip() == 'CM'][['TXN_REF', 'BANKREFNO','DEBIT']].copy()

            # Extract TW (transaction) rows
            tw_df = df[df['TXNCODE'].str.strip() == 'TW'].copy()

            if tw_df.empty:
                logger.info(f"No 'TW' transaction data found in {self.file_path}")
                return

            # Apply transformations
            ##tw_df['rccpkey'] = tw_df['POSTINGDATE'].apply(DateConverter.convert_to_yyyymmdd)
            tw_df['rccpkey'] = (
                pd.to_datetime(tw_df['POSTINGDATE'] + ' ' + tw_df['TIME'],
                            format='%d/%m/%Y %H:%M:%S')
                .dt.strftime('%Y%m%d%H%M')
            )            
            tw_df['rccservice'] = 'BAYDEPOSIT'
            tw_df['rccamount'] = tw_df['DEBIT'].apply(self._convert_amount_value)
            tw_df['source_date'] = tw_df['POSTINGDATE'].apply(DateConverter.convert_to_yyyymmdd)
                       
            # Merge TW with corresponding CM fee on TXN_REF
            tw_df = tw_df.merge(
                fee_df,
                on=['TXN_REF', 'BANKREFNO'],
                how='left',
                suffixes=('', '_fee')
            )

            # Fill NaN if no CM row found
            tw_df['rccfee'] = tw_df['DEBIT_fee'].fillna(0)
            # Convert values if needed
            tw_df['rccfee'] = tw_df['rccfee'].apply(self._convert_amount_value)
            
            output_filename = f"{self.output_prefix}_{self.date}.csv"
            output_path = os.path.join(os.path.dirname(self.file_path), output_filename)
            self.save_as_csv(tw_df, output_path)

        except Exception as e:
            logger.error(f"Error processing file {self.file_path} with BAYStatementProcessor: {e}", exc_info=True)
            raise


class BBLStatementProcessor(BaseProcessor):
    """Processes BBL statement Excel files."""

    def __init__(self, file_path: str, source_config: Dict):
        super().__init__(file_path, source_config.get("date_pattern", ""))
        self.source_config = source_config
        self.output_prefix = self.source_config.get("outbound_prefix","bank_bbl_stm")

        name = os.path.basename(file_path)
        match_new = re.search(r'_(\d{14})\.', name)
        if match_new:
            timestamp_str = match_new.group(1)
            date_part_str = timestamp_str[:8]
            try:
                # The report is for the previous day
                report_date = datetime.strptime(date_part_str, '%Y%m%d')
                self.date = report_date.strftime('%Y%m%d')
                logger.info(f"Extracted date {self.date} from new KTB filename format: {name}")
            except ValueError:
                logger.warning(f"Could not parse date from {name}, falling back to default.")
                self.date = datetime.utcnow().strftime('%Y%m%d')
        else:
            # Fallback to the original logic for older formats
            match_old = re.search(r'(\d{8})', name)
            if match_old:
                self.date = match_old.group(1)
            else:
                self.date = datetime.utcnow().strftime('%Y%m%d')
                logger.warning(f"No YYYYMMDD date found in filename {file_path}. Using current date: {self.date}")

    def process(self):
        """
        Processes the BBL statement Excel file by pairing transaction and fee rows.
        """
        try:
            logger.info(f"Processing file with BBLStatementProcessor: {self.file_path}")

            df = pd.read_excel(self.file_path, skiprows=5)
            df.columns = df.columns.str.strip()

            processed_data = []

            for i, row in df.iterrows():
                description = str(row.get('Description', '')).strip()

                if description == 'MISCELLANEOUS PAYMENT':
                    transaction_data = row.to_dict()

                    transaction_data['rccpkey'] = DateConverter.convert_to_yyyymmdd(row['Value Date'])
                    transaction_data['rccservice'] = 'BBLDEPOSIT'
                    transaction_data['source_date'] = DateConverter.convert_to_yyyymmdd(row['Value Date'])
                    transaction_data['rccamount'] = self._convert_amount_value(row['Debit'])

                    rccfee = "0.00"
                    if (i + 1) < len(df):
                        next_row = df.iloc[i + 1]
                        next_description = str(next_row.get('Description', '')).strip()
                        if next_description == 'CASH SERVICE FEE':
                            rccfee = self._convert_amount_value(next_row['Debit'])
                    
                    transaction_data['rccfee'] = rccfee

                    processed_data.append(transaction_data)

            if not processed_data:
                logger.info(f"No 'MISCELLANEOUS PAYMENT' transactions found in {self.file_path}")
                return

            processed_df = pd.DataFrame(processed_data)

            output_filename = f"{self.output_prefix}_{self.date}.csv"
            output_path = os.path.join(os.path.dirname(self.file_path), output_filename)
            self.save_as_csv(processed_df, output_path)

        except Exception as e:
            logger.error(f"Error processing file {self.file_path} with BBLStatementProcessor: {e}", exc_info=True)
            raise


class KerryCODProcessor(BaseProcessor):
    """Processor for Kerry COD Excel files."""

    def __init__(self, file_path: str, source_config: Dict):
        super().__init__(file_path, source_config.get("date_pattern", ""))
        self.source_config = source_config
        self.date = datetime.utcnow().strftime('%Y%m%d')

    def process(self):
        """
        Processes the Kerry COD Excel file to transform it into the reconcile format.
        """
        try:
            logger.info(f"Processing file with KerryCODProcessor: {self.file_path}")

            ##sheet_name = "COD" --> use sheet_name=1 instead for sheet position
            df = pd.read_excel(self.file_path, sheet_name=1)

            # Create the new columns based on user's requirements
            df['rccpkey'] = df['Waybill No.']
            df['rccservice'] = 'KEX'
            df['rccfee'] = '0.00'
            df['source_date'] = df['Pick-Up Date'].apply(DateConverter.convert_to_yyyymmdd)
            df['batch_date'] = self.date

            # Use the base method to create a standardized 'rccamount' column
            self._add_converted_amount_column(df, "COD Amount", "rccamount")

            # Save the transformed data to a new CSV file
            output_prefix_filename = self.source_config.get("outbound_prefix", "speed_cod_kex")
            output_filename = f"{output_prefix_filename}_{self.date}.csv"
            output_path = os.path.join(os.path.dirname(self.file_path), output_filename)
            self.save_as_csv(df, output_path)
            logger.info(f"Successfully processed and saved {self.file_path} to {output_path}")

        except Exception as e:
            logger.error(f"Error processing file {self.file_path} with KerryCODProcessor: {e}", exc_info=True)
            raise


class KerryDropOffProcessor(BaseProcessor):
    """Processor for Kerry DROP-OFF Excel files."""

    def __init__(self, file_path: str, source_config: Dict):
        super().__init__(file_path, source_config.get("date_pattern", ""))
        self.source_config = source_config
        self.date = datetime.utcnow().strftime('%Y%m%d')

    def process(self):
        """
        Processes the Kerry DROP-OFF Excel file to transform it into the reconcile format.
        """
        try:
            logger.info(f"Processing file with KerryDropOffProcessor: {self.file_path}")

            ###sheet_name = "图表1 202509011454" --> sheet 2 then sheet_name=1
            df = pd.read_excel(self.file_path, sheet_name=1)

            # Create the new columns based on user's requirements
            df['rccpkey'] = df['consignment_no'].str.strip()
            df['rccamount'] = '0.00'
            df['rccfee'] = '0.00'
            df['rccservice'] = 'KEX-DO'
            df['source_date'] = df['Drop_Date'].apply(DateConverter.convert_to_yyyymmdd)
            df['batch_date'] = self.date

            # Save the transformed data to a new CSV file
            output_prefix_filename = self.source_config.get("outbound_prefix", "speed_kexdo")
            output_filename = f"{output_prefix_filename}_{self.date}.csv"
            output_path = os.path.join(os.path.dirname(self.file_path), output_filename)
            self.save_as_csv(df, output_path)
            logger.info(f"Successfully processed and saved {self.file_path} to {output_path}")

        except Exception as e:
            logger.error(f"Error processing file {self.file_path} with KerryDropOffProcessor: {e}", exc_info=True)
            raise


class SpeedKerryDropOffProcessor(BaseProcessor):
    """Processor for SPD Kerry Drop-off CSV files -> reconcile format."""

    def __init__(self, file_path: str, source_config: Dict):
        super().__init__(file_path, source_config.get("date_pattern", ""))
        self.source_config = source_config
        # Derive date for output naming from filename (YYYYMMDD) or fallback to UTC today
        name = os.path.basename(file_path).rsplit(".", 1)[0]
        matches = re.findall(r"\d{8}", name)
        if matches:
            self.date = matches[-1]
        else:
            self.date = datetime.utcnow().strftime("%Y%m%d")
            logger.warning(f"No YYYYMMDD date found in filename {file_path}. Using current date: {self.date}")

    def process(self):
        """
        Transform SPD Kerry Drop-off CSV into reconcile file format:
          - rccpkey   = tracking_code
          - rccamount = "0.00"
          - rccfee    = "0.00"
          - source_date = trns_date (converted to YYYYMMDD)
          - rccservice = 'KEX-DO'
        Keeps all original columns and appends the reconcile columns.
        """
        try:
            logger.info(f"Processing file with SpeedKerryDropOffProcessor: {self.file_path}")

            # Read CSV as strings
            df = pd.read_csv(self.file_path, dtype=str)

            # Core reconcile fields
            df["rccpkey"] = df["tracking_code"]
            df["rccservice"] = "KEX-DO"
            df["rccamount"] = "0.00"
            df["rccfee"] = "0.00"
            df["source_date"] = df["trns_date"].apply(DateConverter.convert_to_yyyymmdd)
            df["batch_date"] = self.date

            # Save output
            output_prefix = self.source_config.get("outbound_prefix", "local_speed_kexdo")
            output_filename = f"{output_prefix}_{self.date}.csv"
            output_path = os.path.join(os.path.dirname(self.file_path), output_filename)
            self.save_as_csv(df, output_path)
            logger.info(f"Successfully processed and saved {self.file_path} to {output_path}")

        except Exception as e:
            logger.error(f"Error processing file {self.file_path} with SpeedKerryDropOffProcessor: {e}", exc_info=True)
            raise


class KerryInvoiceExternalProcessor(BaseProcessor):
    """Processor for Kerry External Invoice Detail Excel files."""

    def __init__(self, file_path: str, source_config: Dict):
        super().__init__(file_path, source_config.get("date_pattern", ""))
        self.source_config = source_config

        name = f'20{os.path.basename(file_path).rsplit("_", 3)[2]}'
        
        match = re.search(r'(\d{8})', name)
        if match:
            self.date = name
        else:
            self.date = datetime.utcnow().strftime('%Y%m%d')
            logger.warning(f"No YYYYMMDD date found in filename {file_path}. Using current date: {self.date}")        
        
        ###self.date = datetime.utcnow().strftime('%Y%m%d')

    def process(self):
        """
        Processes the Kerry External Invoice Detail Excel file to transform it into the reconcile format.
        """
        try:
            logger.info(f"Processing file with KerryInvoiceExternalProcessor: {self.file_path}")

            # 1. Extract 'Customer ID:' from row 8 (index 7)
            customer_id_df = pd.read_excel(self.file_path, header=None, skiprows=7, nrows=1, usecols=[2])
            kex_account = customer_id_df.squeeze()
            ##print(kex_account)

            # 2. Read two-row headers (rows 12 and 13, indices 11 and 12)
            #header_rows_df = pd.read_excel(self.file_path, header=None, skiprows=11, nrows=2)
            
            # Combine the two header rows to create meaningful column names
            # Fill NaN in the first row with empty strings, then concatenate with the second row
            # This assumes the primary header is in the first row and sub-headers are in the second.
            # Adjust this logic if the header structure is different (e.g., second row fills blanks in first).
            combined_headers = ['No.', 'Invoice Number', 'Consignment', 'Tax Type', 'Charge Type',
                 'Charge Amount (VAT exclusive)', 'Discount Amount (VAT exclusive)', 'Net Amount (VAT exclusive)', 
                 'Pickup Date', 'Delivery Date', 'Recipience Name', 'Status', 'WEIGHT Act.', 'Dim.', 'Chg.', 'Pkgs.', 
                 'Ser. Type', 'Origin Province', 'Origin ZipCode', 'Origin Network Code', 'Destination Province', 'Destination ZipCode', 'Destination Network Code']
            # for col_idx in range(len(header_rows_df.columns)):
            #     header1 = str(header_rows_df.iloc[0, col_idx]).strip() if pd.notna(header_rows_df.iloc[0, col_idx]) else ''
            #     header2 = str(header_rows_df.iloc[1, col_idx]).strip() if pd.notna(header_rows_df.iloc[1, col_idx]) else ''
                
            #     if header1 and header2 and header1 != header2: # If both exist and are different, combine
            #         combined_headers.append(f"{header1} {header2}".strip())
            #     elif header1: # If only first exists
            #         combined_headers.append(header1)
            #     elif header2: # If only second exists
            #         combined_headers.append(header2)
            #     else: # If both are empty/NaN
            #         combined_headers.append(f"Unnamed: {col_idx}") # Placeholder for empty columns

            # 3. Read the main data starting from row 14 (index 13) with no header
            df = pd.read_excel(self.file_path, header=None, skiprows=13)
            df.columns = combined_headers # Assign the combined headers
            #print(combined_headers)

            # Keep all original columns and add new columns
            # Ensure column names match the combined headers
            df['source_date'] = df['Pickup Date'].apply(lambda x: DateConverter.convert_to_yyyymmdd(x) if pd.notna(x) else self.date)
            ##df['rccservice'] = 'KEX-INV-' + df['Charge Type'].astype(str).apply(lambda x: x.split(' ')[0] if pd.notna(x) else '')
            df['rccservice'] = 'KEX-INV-' + df['Charge Type'].str.split().str[0].fillna('').str.upper()

            self._add_converted_amount_column(df, "Charge Amount (VAT exclusive)", "rccamount")
            df['rccpkey'] = df['Consignment'].astype(str).str.strip()
            df['rccfee'] = '0.00'
            df['batch_date'] = self.date
            df['kex_account'] = kex_account

            # Save output
            output_prefix_filename = self.source_config.get("outbound_prefix", "speed_kex_inv")
            output_filename = f"{output_prefix_filename}_{kex_account}_{self.date}.csv"
            output_path = os.path.join(os.path.dirname(self.file_path), output_filename)
            self.save_as_csv(df, output_path)
            logger.info(f"Successfully processed and saved {self.file_path} to {output_path}")

        except Exception as e:
            logger.error(f"Error processing file {self.file_path} with KerryInvoiceExternalProcessor: {e}", exc_info=True)
            raise


class SpeedKerryExpressFreightProcessor(BaseProcessor):
    """Processor for SPD Kerry Express Freight CSV files -> reconcile format."""

    def __init__(self, file_path: str, source_config: Dict):
        super().__init__(file_path, source_config.get("date_pattern", ""))
        self.source_config = source_config
        # Derive date for output naming from filename (YYYYMMDD) or fallback to UTC today
        name = os.path.basename(file_path).rsplit(".", 1)[0]
        matches = re.findall(r"\d{8}", name)
        if matches:
            self.date = matches[-1]
        else:
            self.date = datetime.utcnow().strftime("%Y%m%d")
            logger.warning(f"No YYYYMMDD date found in filename {file_path}. Using current date: {self.date}")

    def process(self):
        """
        Transforms SPD Kerry Express Freight CSV into reconcile file format.
        """
        try:
            logger.info(f"Processing file with SpeedKerryExpressFreightProcessor: {self.file_path}")

            df = pd.read_csv(self.file_path, dtype=str)

            # Add reconcile columns
            df["rccpkey"] = df["tracking_code"]
            df["rccservice"] = "KEX-INV-FREIGHT"
            self._add_converted_amount_column(df, "freight_charge", "rccamount")
            df["rccfee"] = "0.00"
            df["source_date"] = df["trns_date"].apply(DateConverter.convert_to_yyyymmdd)
            df["batch_date"] = self.date

            # Save output file
            output_prefix = self.source_config.get("outbound_prefix", "local_speed_kex_inv_freight")
            output_filename = f"{output_prefix}_{self.date}.csv"
            output_path = os.path.join(os.path.dirname(self.file_path), output_filename)
            self.save_as_csv(df, output_path)
            logger.info(f"Successfully processed and saved {self.file_path} to {output_path}")

        except Exception as e:
            logger.error(f"Error processing file {self.file_path} with SpeedKerryExpressFreightProcessor: {e}", exc_info=True)
            raise

class SpeedKerryExpressFuelProcessor(BaseProcessor):
    """Processor for SPD Kerry Express Fuel CSV files -> reconcile format."""

    def __init__(self, file_path: str, source_config: Dict):
        super().__init__(file_path, source_config.get("date_pattern", ""))
        self.source_config = source_config
        # Derive date for output naming from filename (YYYYMMDD) or fallback to UTC today
        name = os.path.basename(file_path).rsplit(".", 1)[0]
        matches = re.findall(r"\d{8}", name)
        if matches:
            self.date = matches[-1]
        else:
            self.date = datetime.utcnow().strftime("%Y%m%d")
            logger.warning(f"No YYYYMMDD date found in filename {file_path}. Using current date: {self.date}")

    def process(self):
        """
        Transforms SPD Kerry Express Fuel CSV into reconcile file format.
        """
        try:
            logger.info(f"Processing file with SpeedKerryExpressFuelProcessor: {self.file_path}")

            df = pd.read_csv(self.file_path, dtype=str)

            # Add reconcile columns
            df["rccpkey"] = df["tracking_code"]
            df["rccservice"] = "KEX-INV-FUEL"
            self._add_converted_amount_column(df, "fuel_surcharge", "rccamount")
            df["rccfee"] = "0.00"
            df["source_date"] = df["trns_date"].apply(DateConverter.convert_to_yyyymmdd)
            df["batch_date"] = self.date

            # Save output file
            output_prefix = self.source_config.get("outbound_prefix", "local_speed_kex_inv_fuel")
            output_filename = f"{output_prefix}_{self.date}.csv"
            output_path = os.path.join(os.path.dirname(self.file_path), output_filename)
            self.save_as_csv(df, output_path)
            logger.info(f"Successfully processed and saved {self.file_path} to {output_path}")

        except Exception as e:
            logger.error(f"Error processing file {self.file_path} with SpeedKerryExpressFuelProcessor: {e}", exc_info=True)
            raise


class CSVProcessor(BaseProcessor):
    """Processes CSV files for merchant e-payment data."""

    ##def __init__(self, file_path: str, date_pattern: str, csv_data_option: str, source_config: str):
    def __init__(self, file_path: str, source_config):
        """Initialize CSV processor with file path and date pattern."""
        self.file_path = file_path
        self.path_name = os.path.dirname(file_path)
        ##self.date = DateTransformer.extract_from_filename(file_path, source_config.get('date_pattern', ''))
        # Remove the file extension
        name = file_path.rsplit(".", 1)[0]        

        # Find all matches for YYYYMMDD (8 digits) or DD.MM.YY (2.2.2)
        matches = re.findall(r"\d{8}|\d{2}\.\d{2}\.\d{2}", name)
        # If no matches found, return None (could raise an error if preferred)
        if not matches:
            raise ValueError(
                f"File {self.file_path} date {matches} in filename does not matched with date format"
            )
        # Take the last match, as the date is typically near the end
        date_str = matches[-1]        

        #self.date = DateTransformer.extract_date(file_path)
        #self.date = DateConverter.convert_to_yyyymmdd(date_str)
        self.date = date_str
        
        self.csv_data_option = source_config.get("csv_data_option", "local")
        self.csv_option = self._parse_file_options(source_config)
        self.outbound_prefix = source_config.get("outbound_prefix", "")
        ###logger.info(f"CSV option {self.csv_option}")
        logger.info(f"Initialized CSV processor for {file_path}")

    # CSV Configuration Options
    def _process_delimiter(self, value):
        """Convert 'tab' or escaped characters like \\t to actual tab"""
        if value.lower() == "tab":
            return "\t"
        elif value.lower() == "pipe":
            return "|"
        return bytes(value, "latin-1").decode("unicode_escape")

    def _parse_file_options(self, source_config):
        # Parse delimiter with a default of comma
        delimiter = self._process_delimiter(source_config.get("csv_delimiter", ","))

        # Parse columns, expecting a string or None
        columns_str = source_config.get("csv_usecols", None)
        if isinstance(columns_str, str):
            # Split and clean column names, filtering out empty entries
            columns = [col.strip() for col in columns_str.split(",") if col.strip()]
        else:
            columns = None  # None means use all columns in CSV parsing

        # Parse other options with appropriate defaults
        has_header = source_config.get("csv_header", True)  # Assume header by default
        header_columns = source_config.get("csv_header_columns", None)  # Optional
        skip_header = source_config.get(
            "csv_skip_header", 0
        )  # No lines skipped by default
        skip_footer = source_config.get(
            "csv_skip_footer", 0
        )  # No lines skipped by default

        # Return the parsed options as a dictionary
        return {
            "delimiter": delimiter,
            "has_header": has_header,
            "header_columns": header_columns,
            "skip_header": skip_header,
            "skip_footer": skip_footer,
            "usecols": columns,
        }

    # Function to read CSV with configurable options
    # def _read_csv_with_options(self, header_columns=None, delimiter=',', has_header=True, skip_header=0, skip_footer=0, usecols=None) ->pd.DataFrame:
    def _read_csv_with_options(
        self, has_header=True, header_columns=None
    ) -> pd.DataFrame:
        engine = "python"
        if not has_header and header_columns is None:
            raise ValueError("header_columns must be provided when has_header is False")

        names = None if has_header else header_columns.split(",")
        header = 0 if has_header else None

        try:
            return pd.read_csv(
                self.file_path,
                quotechar='"',
                dtype="str",
                delimiter=",",
                header=header,
                names=names,
                engine=engine,
            )
        except pd.errors.EmptyDataError:
            raise ValueError(
                f"File {self.file_path} is empty after applying skip_header and skip_footer"
            )

    # Function to read CSV with configurable options
    def _read_csv_with_options2(
        self,
        delimiter=",",
        has_header=True,
        header_columns=None,
        skip_header=0,
        skip_footer=0,
        usecols=None,
    ):
        engine = "python"
        ##if skip_footer > 0 else None
        if not has_header and header_columns is None:
            raise ValueError("header_columns must be provided when has_header is False")

        names = None if has_header else header_columns.split(",")
        header = 0 if has_header else None

        ###logger.info(f"delimiter: {delimiter}")

        try:
            df = pd.read_csv(
                self.file_path,
                quotechar='"',
                delimiter=delimiter,
                header=header,
                dtype="str",
                names=names,
                skiprows=skip_header,
                skipfooter=skip_footer,
                engine=engine,
                usecols=usecols,
            )
            ###encoding='TIS-620'
            return df
        except pd.errors.EmptyDataError:
            raise ValueError(
                f"File {self.file_path} is empty after applying skip_header and skip_footer"
            )

    def _read_csv(self) -> pd.DataFrame:
        return pd.read_csv(self.file_path, engine="python", dtype="str")

    def _process_tmnwallet(self, df: pd.DataFrame) -> None:
        # df = df.replace("'", "", regex=True)
        ##df['source_date'] = df['วันที่เวลา'].apply(DateTransformer.iso_to_yyyymmdd2)
        ###df["source_date"] = df["SETTLEMENT_TIME"].apply(DateTransformer.iso_to_yyyymmdd3)
        df["source_date"] = df["SETTLEMENT_TIME"].apply(DateConverter.convert_to_yyyymmdd)
        
        # Define the list of conditions
        conditions = [
            (df["MERCHANT_ID"] == "010000000000406976074") & (df["SHOP_ID"] == "300000000000000122744"),
            (
                ((df["MERCHANT_ID"] == "010000000000327726579") & (df["SHOP_ID"] == "300000000000000058613")) |
                ((df["MERCHANT_ID"] == "010000000000327694573") & (df["SHOP_ID"] == "300000000000000058611"))
            ),
            df["PARTNER_SHOP_ID"].str.startswith("M", na=False)
        ]

        # Define the list of corresponding choices
        choices = [
            "easy",
            "pos",
            "jsk_vending"
        ]

        # Apply the conditions using np.select
        df["rccservice"] = np.select(conditions, choices, default="unknown")

        # Assign rccpkey based on rccservice
        df["rccpkey"] = np.where(
            df["rccservice"].isin(["jsk_vending", "easy"]),
            df["PARTNER_TRANSACTION_ID"].astype(str) + "|" + df["rccservice"],
            df["PARTNER_TRANSACTION_ID"].astype(str) + "|" + df["TRANSACTION_ID"].astype(str) + "|" + df["rccservice"]
        )        

        df_filtered = df[df["rccservice"] != "unknown"]
        self._add_converted_amount_column(df_filtered, "AMOUNT", "rccamount")

        self._save_csv(
            df_filtered,
            ##os.path.join(self.path_name, f"merchant_epay_tmnwallet_{self.date}.csv"),
            os.path.join(self.path_name, f"{self.outbound_prefix}_{self.date}.csv"),

        )

    def _process_local(self, df: pd.DataFrame) -> None:
        """Process local merchant e-payment CSV data."""
        # Filter out the unwanted site-spo_food_court rows, comment out this line to rec with external source.
        ##df = df[df["payment_origin"] != "site-spo_food_court"] 
        ##df["source_date"] = df["payment_gw_when"].apply(DateTransformer.iso_to_yyyymmdd)
        df["source_date"] = df["payment_gw_when"].apply(DateConverter.convert_to_yyyymmdd)

        filters = {
            "pos": "device-POS",
            "bpos": "site-bpos",
            "thaivan": (
                "site-life_four_cuts",
                "site-thaivan_l4c_kdecor",
                "site-thaivan_l4c_kvkk_co_ltd",
                "site-thaivan_l4c_tk86",
            ),
            "easy": "site-easy",
            "gameonline": ("site-sbm_asphere", "site-sbm_game_square"),
            "jsk_vending": "site-sbm_jsk_vending",
        }

        def get_suffix(ref):
            for prefix, value in filters.items():
                if isinstance(value, str) and ref.startswith(value):
                    return prefix
                elif isinstance(value, tuple) and any(ref.startswith(v) for v in value):
                    return prefix
            return "unknown"

        df["rccservice"] = df["payment_origin"].apply(get_suffix)

        # modified 2025-05-06 by nawatsapon.t to update Refine rccservice for wechat, alipay

        ## comment old
        # Refine rccservice for 'easy' cases based on payment_sof
        # mask_easy = df["rccservice"] == "easy"
        # df.loc[mask_easy & (df["payment_sof"] == "alipay-scb2"), "rccservice"] = (
        #     "alipay"
        # )
        # df.loc[mask_easy & (df["payment_sof"] == "wechat-scb2"), "rccservice"] = (
        #     "wechat"
        # )
        ## add new
        # df["rccservice"] = np.where(
        #     df["payment_sof"] == "alipay-scb2", "alipay-scb2",
        #     np.where(
        #         df["payment_sof"] == "wechat-scb2", "wechat-scb2",
        #         df["rccservice"]  # 👈 keep original value if no match
        #     )
        # )

        df["rccservice"] = np.where(
            (df["payment_sof"].isin(['alipay-scb2', 'wechat-scb2'])) & (df["rccservice"] != 'unknown'),
            df["payment_sof"],
            np.where(
            (df["payment_sof"].isin(['alipay-scb2', 'wechat-scb2'])) & (df["payment_origin"] == 'site-spo_food_court'),
            df["payment_sof"],
            np.where( df["payment_terminal_id"].str.startswith("POS"), 'pos'
            ,
            df["rccservice"] # 👈 keep original value if no match
            )
           )
        )        

        # df["rccservice"] = np.where(
        #     (df["payment_sof"].isin(['alipay-scb2', 'wechat-scb2'])) & (df["rccservice"] != 'unknown'),
        #     df["payment_sof"],
        #     df["rccservice"]  # keep original value if no match
        # )        
                ## End modified : update Refine rccservice for wechat, alipay

        # Rows where rccservice is 'easy' and payment_sof is neither 'alipay' nor 'wechat' remain 'easy'

        ###df['rccpkey'] = df['payment_ref1'].astype(str) + '|' + df['payment_ref2'].astype(str) + '|' + df['rccservice']
        # Create rccpkey
        ###df['rccpkey'] = df['payment_ref1'].astype(str) + '|' + df['payment_ref2'].astype(str) + '|' + df['rccservice']

        # Create rcckey
        df["rccpkey"] = np.where(
            df["rccservice"].isin(["alipay-scb2", "wechat-scb2"]),
            df["payment_ref"].astype(str) + "|" + df["rccservice"],
            np.where(
              (df["rccservice"] == "jsk_vending") & (df["payment_sof"] == "tmnwallet"),
              df["payment_ref"].astype(str) + "|" + df["rccservice"],
            np.where(
                (df["payment_sof"] == "tmnwallet-truemoney")
                & (df["rccservice"].isin(["pos", "easy"])),
                df["payment_ref"].astype(str)
                + "|"
                + df["payment_gw_ref"].astype(str)
                + "|"
                + df["rccservice"],
                np.where(
                    df["rccservice"] == "pos",
                    df["payment_ref1"].astype(str) + "|" + df["rccservice"],
                    df["payment_ref1"].astype(str)
                    + "|"
                    + df["payment_ref2"].astype(str)
                    + "|"
                    + df["rccservice"],
                ),
            ),
            ),
        )

        df_filtered = df[df["rccservice"] != "unknown"]

        self._add_converted_amount_column(df_filtered, "payment_amount", "rccamount")

        self._save_csv(
            df_filtered,
            os.path.join(self.path_name, f"{self.outbound_prefix}_{self.date}.csv"),
        )

    def _process_wechat(self, df: pd.DataFrame) -> None:
        """Process wechat alipay merchant e-payment CSV data."""
        ##print (df.info)
        ##df["source_date"] = df["TXN_Date"].apply(DateTransformer.thai_to_yyyymmdd)
        df["source_date"] = df["TXN_Date"].apply(DateConverter.convert_to_yyyymmdd)
        
        ## Modified 2025-05-06 by nawatsapon.t: update rccservice key for alipay, wechat
        ## df["rccservice"] = np.where(df["Product"] == "Alipay+", "alipay", "wechat")
        df["rccservice"] = np.where(df["Product"] == "Alipay+", "alipay-scb2", "wechat-scb2")
        ## End Modified: update rccservice key for alipay, wechat
        df["rccpkey"] = df["Transaction_ID"].astype(str) + "|" + df["rccservice"]

        df_filtered = df[df["rccservice"] != "unknown"]

        self._add_converted_amount_column(df_filtered, "Amount", "rccamount")
        self._save_csv(
            df_filtered,
            os.path.join(
                self.path_name, f"{self.outbound_prefix}_{self.date}.csv"
            ),
        )

    def _process_spd_cod_local(self, df: pd.DataFrame) -> None:
        """Process local spd cod daily CSV data."""
        # Create source_date from the sale_date column
        df['source_date'] = df['sale_date'].apply(DateConverter.convert_to_yyyymmdd)

        # Use the 'courier' column for rccservice
        df['rccservice'] = df['courier']
        df["rccservice"] = np.where(
            df["courier"].isin(['FLSESP', 'FLEB']), 'FLASH',
            np.where(
            df["courier"].isin(['KEXEASY', 'KEXFRUIT', 'KERRY', 'KEXF']), 'KEX',
            df["courier"] # 👈 keep original value if no match
            )
        )

        # Create a unique reconciliation key from tracking_code and rccservice
        df['rccpkey'] = df['tracking_code'].astype(str) 

        # Convert the cod_amount to the standardized rccamount
        self._add_converted_amount_column(df, "cod_amount", "rccamount")

        # Filter out rows where rccservice might be unknown, as a safeguard
        ##df_filtered = df[df["rccservice"] == "POSTSABUY"]
        df_filtered = df[df['rccservice'].isin(['POSTSABUY', 'FLASH', 'KEX'])]

        # Save the processed data to a new CSV file
        self._save_csv(
            df_filtered,
            os.path.join(self.path_name, f"{self.outbound_prefix}_{self.date}.csv"),
        )        

    ##    def process(self, header: bool, header_columns: str) -> None:
    def process(self) -> None:
        """Execute processing based on the specified method."""
        try:
            if self.csv_data_option == "local":
                # df = self._read_csv()
                ##df = self._read_csv_with_options(header, header_columns)
                df = self._read_csv_with_options2(**self.csv_option)
                self._process_local(df)
            elif self.csv_data_option == "tmnwallet":
                df = self._read_csv_with_options2(**self.csv_option)
                self._process_tmnwallet(df)
            elif self.csv_data_option == "wechat_alipay":
                ##df = self._read_csv_with_options(header, header_columns)
                df = self._read_csv_with_options2(**self.csv_option)
                self._process_wechat(df)
            elif self.csv_data_option == "spd_cod_local":
                df = self._read_csv_with_options2(**self.csv_option)
                self._process_spd_cod_local(df)                
            # if header:
            #     df = self._read_csv()
            #     self._process_local(df)
            # else:
            #     df = self._read_csv_with_options(header, header_columns)
            #     self._process_wechat(df)
            logger.info(f"Completed processing {self.file_path}")
        except Exception as e:
            logger.error(f"Processing failed for {self.file_path}: {e}")
            raise

### add new function to support move source file to archive after complted process
def move_local_to_archive(src_dir: str,
                          filenames: list[str],
                          archive_dir: str | None = None) -> None:
    """
    Move processed files from *src_dir* to *archive_dir* on the local disk.

    Parameters
    ----------
    src_dir : str
        Folder where the freshly‑processed files currently live.
    filenames : list[str]
        List of file names (not full paths) to archive.
    archive_dir : str | None, default None
        Destination folder. If None, falls back to <src_dir>/archive.

    Archive folder is created automatically if it does not exist.
    Existing files with the same name are overwritten.
    """
    if not archive_dir:
        archive_dir = os.path.join(src_dir, "archive")
    
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
        logger.info(f"Created local archive directory: {archive_dir}")

    for filename in filenames:
        src_path = os.path.join(src_dir, filename)
        dest_path = os.path.join(archive_dir, filename)
        try:
            shutil.move(src_path, dest_path)
            logger.info(f"Archived local file: {src_path} to {dest_path}")
        except Exception as e:
            logger.error(f"Failed to move {src_path} to {dest_path}: {e}")

def main(service_name: str = "merchant_e_payment") -> None:
    """Process all configured sources for the specified service."""
    config = load_config()
    service_config = next(s for s in config["services"] if s["name"] == service_name)
    sftp_config = next(s for s in config["sftp"] if s["name"] == "PROD")
    sftp = SFTPManager(
        sftp_config["server"]["host"],
        sftp_config["server"]["port"],
        sftp_config["server"]["username"],
        sftp_config["server"].get("password"),
    )

    source_keys = [key for key in service_config if key.endswith("_source")]
    logger.info(f"Start processing {source_keys}")
    for source_key in source_keys:
        source = service_config[source_key]

        ##files = sftp.download_files(source['sftp_inbound_folder'], source['input_file_pattern'],
        ##                          source['reconcile_input_folder'])

        files = sftp.download_files(
            source["sftp_inbound_folder"],
            source["input_file_pattern"],
            source["reconcile_input_folder"],
            source.get("zip_file", False),
            source.get("unzip_password", None),
            source.get("sftp_archive_folder", None),
        )

        if source["type"]== "bank_statement_excel":
            for file_name in files:
                file_path = os.path.join(source["reconcile_input_folder"], file_name)
                processor = BankStatementProcessor(file_path, source)
                processor.process()                    
        elif source["type"]  == "spd_cod_thaipost_xls":  # Extermal Postsabuy COD source
            for file_name in files:
                file_path = os.path.join(source["reconcile_input_folder"], file_name)
                processor = PostsabuyCODProcessor(file_path, source)
                processor.process()            
        elif source["type"] == "bank_agent_txn":
            for file_name in files:
                file_path = os.path.join(source["reconcile_input_folder"], file_name)
                processor = BankAgentTxnProcessor(file_path, source)
                processor.process()           
        elif source["type"] == "spd_cod_flash":  # External Flash COD source
            for file_name in files:
                file_path = os.path.join(source["reconcile_input_folder"], file_name)
                processor = FlashCODProcessor(file_path, source)
                processor.process()       
        elif source["type"] == "spd_cod_kerry_xls":  # External KEX COD source
            for file_name in files:
                file_path = os.path.join(source["reconcile_input_folder"], file_name)
                processor = KerryCODProcessor(file_path, source)
                processor.process()       
        elif source["type"] == "spd_do_kerry_xls":  # External KEX drop-off sorurce
            for file_name in files:
                file_path = os.path.join(source["reconcile_input_folder"], file_name)
                processor = KerryDropOffProcessor(file_path, source)
                processor.process()       
        elif source["type"] == "spd_kexdo_txn":  # Internal KEX drop-off sorurce
            for file_name in files:
                file_path = os.path.join(source["reconcile_input_folder"], file_name)
                processor = SpeedKerryDropOffProcessor(file_path, source)
                processor.process()       
        elif source["type"] == "spd_kerry_inv_xlsx":  # External KEX Invoice source
            for file_name in files:
                file_path = os.path.join(source["reconcile_input_folder"], file_name)
                processor = KerryInvoiceExternalProcessor(file_path, source)
                processor.process()                     
        elif source["type"] == "spd_kerry_inv_fuel":  # Internal KEX Invoice  Fuel source
            for file_name in files:
                file_path = os.path.join(source["reconcile_input_folder"], file_name)
                processor = SpeedKerryExpressFuelProcessor(file_path, source)
                processor.process()
        elif source["type"] == "spd_kerry_inv_freight":  # Internal KEX Invoice  Freight source
            for file_name in files:
                file_path = os.path.join(source["reconcile_input_folder"], file_name)
                processor = SpeedKerryExpressFreightProcessor(file_path, source)
                processor.process()                      
        elif source["type"] == "spd_flash":  # Internal Flash COD parcel source
            for file_name in files:
                file_path = os.path.join(source["reconcile_input_folder"], file_name)
                processor = FlashProcessor(file_path, source)
                processor.process()
        elif source["type"] == "spd_txn":  # Internal speed Flash source
            for file_name in files:
                file_path = os.path.join(source["reconcile_input_folder"], file_name)
                processor = SpeedTransactionProcessor(file_path, source)
                processor.process()                                           
        elif source["type"] == "xlsb":
            for file_name in files:
                file_path = os.path.join(source["reconcile_input_folder"], file_name)
                for sheet_config in source["sheets"]:
                    processor = XLSBProcessor(
                        file_path, sheet_config["name"], source["date_pattern"]
                    )
                    processor.process(sheet_config["processor"]) 
        elif source["type"] == "kbankqr22_csv":
            for file_name in files:
                file_path = os.path.join(source["reconcile_input_folder"], file_name)
                processor = KBankQR22Processor(file_path, source)
                processor.process()                                                  
        elif source["type"] == "kbank_statement_csv":
            for file_name in files:
                file_path = os.path.join(source["reconcile_input_folder"], file_name)
                processor = KBankStatementProcessor(file_path, source)
                processor.process()                             
        elif source["type"] == "baac_statement_csv":
            for file_name in files:
                file_path = os.path.join(source["reconcile_input_folder"], file_name)
                processor = BAACStatementProcessor(file_path, source)
                processor.process()                             
        elif source["type"] == "cimb_statement_excel":
            for file_name in files:
                file_path = os.path.join(source["reconcile_input_folder"], file_name)
                processor = CIMBStatementProcessor(file_path, source)
                processor.process()                             
        elif source["type"] == "ktb_statement_excel":
            for file_name in files:
                file_path = os.path.join(source["reconcile_input_folder"], file_name)
                processor = KTBStatementProcessor(file_path, source)
                processor.process()                             
        elif source["type"] == "scb_statement_csv":
            for file_name in files:
                file_path = os.path.join(source["reconcile_input_folder"], file_name)
                processor = SCBStatementProcessor(file_path, source)
                processor.process()                             
        elif source["type"] == "bay_statement_excel":
            for file_name in files:
                file_path = os.path.join(source["reconcile_input_folder"], file_name)
                processor = BAYStatementProcessor(file_path, source)
                processor.process()                             
        elif source["type"] == "bbl_statement_excel":
            for file_name in files:
                file_path = os.path.join(source["reconcile_input_folder"], file_name)
                processor = BBLStatementProcessor(file_path, source)
                processor.process()                             
        elif source["type"] == "csv":
            for file_name in files:
                if file_name.lower().endswith(('.csv','.txt')):
                    logger.info(f"Processing CSV file: {file_name} 📄")
                    file_path = os.path.join(source["reconcile_input_folder"], file_name)
                    ##processor = CSVProcessor(file_path, source['date_pattern'], source['csv_data_option'])
                    processor = CSVProcessor(file_path, source)
                    ##processor.process(source['csv_header'], source['csv_header_columns'])
                    processor.process()
                else:
                    # Optional: Log which files are being skipped
                    logger.info(f"Skipping non-CSV file: {file_name}")

        # --- 20250421: archive the source file copies -------------------
        if not source.get("zip_file", False):
            for file_name in files:
                sftp.move_to_archive(
                    source["sftp_inbound_folder"],
                    file_name,
                    source["sftp_archive_folder"],
                )
        move_local_to_archive(
            src_dir=source["reconcile_input_folder"],
            filenames=files,
            archive_dir=source.get("local_archive_folder")  # optional; falls back to <reconcile_input_folder>/archive
        )

    sftp.close()


if __name__ == "__main__":
    service_name = "merchant_e_payment"
    if len(sys.argv) != 2:
        logger.info("Not found service name input, Use default=merchant_e_payment")
    else:
        service_name = sys.argv[1]
    main(service_name)     
