import re
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        ##logging.FileHandler('/wsol/app/reconcile_inbound/rec_inbound.log'),
        logging.FileHandler('/wsol/app/logs/new_auto_reconcile/rec_inbound.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DateConverter:
    # Thai month abbreviations
    THAI_MONTHS = {
        'ม.ค.': '01', 'ก.พ.': '02', 'มี.ค.': '03', 'เม.ย.': '04',
        'พ.ค.': '05', 'มิ.ย.': '06', 'ก.ค.': '07', 'ส.ค.': '08',
        'ก.ย.': '09', 'ต.ค.': '10', 'พ.ย.': '11', 'ธ.ค.': '12'
    }

    # Full Thai month names
    THAI_MONTHS_FULL = {
        'มกราคม': '01', 'กุมภาพันธ์': '02', 'มีนาคม': '03', 'เมษายน': '04',
        'พฤษภาคม': '05', 'มิถุนายน': '06', 'กรกฎาคม': '07', 'สิงหาคม': '08',
        'กันยายน': '09', 'ตุลาคม': '10', 'พฤศจิกายน': '11', 'ธันวาคม': '12'
    }

    # Combine Thai months
    THAI_MONTHS_ALL = {**THAI_MONTHS, **THAI_MONTHS_FULL}

    # English month abbreviations
    ENGLISH_MONTHS = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
        'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    }

    @staticmethod
    def convert_to_yyyymmdd(date_input) -> str:
        """
        Convert various date formats to YYYYMMDD.

        Supports:
        - Date strings in Day-Month-Year, Month-Day-Year, or Year-Month-Day formats with numeric, English, or Thai months.
        - Excel serial numbers (integers, floats, or numeric strings) starting from 1900-01-01, adjusting for Excel's leap year bug.
        - Ignores time components if present.

        Args:
            date_input: Date string (e.g., "1-Apr-2025", "Apr/01/2025", "2025/04/01") or serial number (e.g., 45749).

        Returns:
            str: Date in 'YYYYMMDD' format, or None if input is invalid.

        Raises:
            ValueError: If the date cannot be parsed or is invalid.
        """
        # Handle None or NaN
        if pd.isna(date_input):
            return None

        # Handle Excel serial numbers
        if (isinstance(date_input, str) and len(date_input) == 12 and date_input.isdigit()):
            return date_input[:8]  # convert 202503081035 Just return the YYYYMMDD part
        elif isinstance(date_input, (int, float)) or (isinstance(date_input, str) and date_input.replace('.', '', 1).isdigit()):
            serial = int(float(date_input))  # Convert to int, handling float or string
            if serial < 1:
                raise ValueError(f"Invalid serial number: {serial}")
            epoch = datetime(1899, 12, 31)
            adjusted_serial = serial - 1 if serial > 59 else serial  # Adjust for Excel leap year bug
            date = epoch + timedelta(days=adjusted_serial)
            return date.strftime('%Y%m%d')

        # Handle date strings
        if not isinstance(date_input, str):
            try:
                return date_input.strftime('%Y%m%d')
            except (AttributeError, ValueError):
                raise ValueError(f"Unsupported input type: {type(date_input)}")
                
        patterns = [
            r"(\d{1,2})[-/](\w+)[-/](\d{4})",      # Day-Month-Year, month as letters (e.g., "1-Apr-2025")
            r"(\w+)[-/](\d{1,2})[-/](\d{4})",      # Month-Day-Year, month as letters (e.g., "Apr/01/2025")
            r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})",  # Numeric date, ambiguous (e.g., "3/19/2025")
            r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})",  # Year-Month-Day (e.g., "2025-04-01")
            r"(\d{1,2})\s*([^\u0000-\u007F]*)\s*(\d{4})",  # Day Month Year with spaces and unicode (e.g., "24 มีนาคม 2025") for thai only use Range: 0E00–0E7F
            r"(\d{1,2})\s+([^\s]+)\s+(\d{4})",
            r"(\d{1,2})[./](\d{1,2})[./](\d{1,2})",  # Day.Month-Year (e.g., "16.04.25")
            r"(\d{1,2})[-/](\d{1,2})[-/](\d{1,2})\s*(\d{2}):(\d{2})",  # Day.Month-Year (e.g., "16-04-25 15:11")
        ]

        for pattern in patterns:
            match = re.match(pattern, date_input)
            if match:
                groups = match.groups()
                if pattern == patterns[0]:  # Day-Month-Year with month as letters
                    day, month, year = groups
                    month_num = DateConverter.get_month_num(month)
                    if month_num is None:
                        continue
                    day = day.zfill(2)
                elif pattern == patterns[1]:  # Month-Day-Year with month as letters
                    month, day, year = groups
                    month_num = DateConverter.get_month_num(month)
                    if month_num is None:
                        continue
                    day = day.zfill(2)
                elif pattern == patterns[2]:  # Numeric date
                    part1, part2, year = groups
                    part1, part2 = int(part1), int(part2)
                    if part2 > 12:  # Likely month-day-year
                        month, day = part1, part2
                    else:  # Assume day-month-year
                        day, month = part1, part2
                    if not (1 <= month <= 12 and 1 <= day <= 31):
                        continue
                    month_num = str(month).zfill(2)
                    day = str(day).zfill(2)
                elif pattern == patterns[3]:  # Year-Month-Day   
                    year, month, day = groups
                    month_num = month.zfill(2)
                    day = day.zfill(2)
                elif pattern == patterns[4]:  # Day Month Year with spaces
                    day, month, year = groups
                    month_num = DateConverter.get_month_num(month)
                    if month_num is None:
                        continue
                    day = day.zfill(2)
                elif pattern == patterns[6]:  # Day.Month.Year
                    day, month, year = groups
                    month_num = str(month).zfill(2)
                    year = f"20{year}"
                    day = day.zfill(2)     
                elif pattern == patterns[7]:  # Day.Month.Year
                    day, month, year, hh, mm = groups
                    month_num = str(month).zfill(2)
                    year = f"20{year}"
                    day = day.zfill(2)                                   
                else:
                    continue

                result = f"{year}{month_num}{day}"
                # Basic validation
                try:
                    datetime.strptime(result, '%Y%m%d')
                    return result
                except ValueError:
                    continue

        raise ValueError(f"Unable to parse date: {date_input}")

    @staticmethod
    def get_month_num(month_str):
        """Convert month string to two-digit number."""
        if any('\u0e00' <= char <= '\u0e7f' for char in month_str):
            return DateConverter.THAI_MONTHS_ALL.get(month_str)
        return DateConverter.ENGLISH_MONTHS.get(month_str)
    
    @staticmethod
    def extract_from_filename(file_path: str, pattern: str) -> str:
        try:
            base_name = os.path.splitext(file_path)[0]
            parts = base_name.split('-') if '-' in base_name else base_name.split('_')
            if len(parts) < 2:
                raise ValueError("Filename does not contain a hyphen or underscore.")
            date_str = parts[-1].strip()
            logger.info(f"extract_from_filename: extracted raw date_str {date_str}")
            return DateConverter.convert_to_yyyymmdd(date_str)
        except Exception as e:
            logger.error(f"Error extracting date from {file_path}: {e}")
            raise

    @staticmethod
    def extract_date(filename: str) -> str:
        name = filename.rsplit('.', 1)[0]
        matches = re.findall(r'\d{8}|\d{2}\.\d{2}\.\d{2}', name)
        if not matches:
            return None
        try:
            return DateConverter.convert_to_yyyymmdd(matches[-1])
        except Exception as e:
            logger.error(f"Failed to convert extracted date from filename: {e}")
            return None

    @staticmethod
    def thai_yymmdd_to_yyyymmdd(thai_date: str) -> str:
        """
        Converts a Thai date in YYMMDD format to YYYYMMDD.
        Assumes the Thai year is in the 2500s.
        """
        if not isinstance(thai_date, str) or len(thai_date) != 6 or not thai_date.isdigit():
            raise ValueError(f"Invalid Thai YYMMDD date format: {thai_date}")
        
        thai_year_yy = int(thai_date[:2])
        month_day = thai_date[2:]
        
        # Convert Thai YY to Gregorian YYYY
        # 2568 -> 2025.  68 -> 2568. 2568 - 543 = 2025
        gregorian_year = 2500 + thai_year_yy - 543
        
        return f"{gregorian_year}{month_day}" 

# Test cases
# print(DateConverter.convert_to_yyyymmdd(45750))              # '20250403'
# print(DateConverter.convert_to_yyyymmdd("45749"))            # '20250402'
# print(DateConverter.convert_to_yyyymmdd("1-Apr-2025"))       # '20250401'
# print(DateConverter.convert_to_yyyymmdd("1/Apr/2025"))       # '20250401'
# print(DateConverter.convert_to_yyyymmdd("Apr/01/2025"))      # '20250401'
# print(DateConverter.convert_to_yyyymmdd("24 มีนาคม 2025"))   # '20250324'
# print(DateConverter.convert_to_yyyymmdd("24 มีนาคม 2025 22:19:54"))  # '20250324'
# print(DateConverter.convert_to_yyyymmdd("3/19/2025 19:44"))  # '20250319'
# print(DateConverter.convert_to_yyyymmdd("2025-04-01 19:44:33.543"))  # '20250401'
# print(DateConverter.convert_to_yyyymmdd("2025/04/01"))       # '20250401'
# print(DateConverter.convert_to_yyyymmdd("16.04.25"))       # '20250416'
# print(DateConverter.convert_to_yyyymmdd("202503081035"))       # 20250308
# print(DateConverter.convert_to_yyyymmdd("16-04-25 15:41"))       # 20250416