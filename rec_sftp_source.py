# Import required modules
import logging
import yaml
import sys
from datetime import date, timedelta
from typing import Dict
from sftp_manager import SFTPManager
import fnmatch

# Configure logging to see info and error messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/wsol/app/logs/new_auto_reconcile/sftp_manager.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)



def load_config(
    config_file: str = "/wsol/app/new_auto_reconcile/inbound/rec_sftp_source.yaml",
) -> Dict:
    # def load_config(config_file: str = 'rec_sftp_source.yaml') -> Dict:
    """Load configuration from a YAML file."""
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_file}.")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def main(service_name: str = "merchant_e_payment") -> None:
    """Process all configured sources for the specified service."""
    config = load_config()
    service_config = next(s for s in config["services"] if s["name"] == service_name)

    source_keys = [key for key in service_config if key.endswith("_source")]
    logger.info(f"Start processing {source_keys}")
    try:
        for source_key in source_keys:
            source = service_config[source_key]

            date_condition = source.get("date_condition", None)

            input_file_pattern = source.get("input_file_pattern")

            source_archive_folder = source.get("source_archive_folder", None)

            if date_condition == "TODAY-1":
                # Calculate yesterday's date
                yesterday = date.today() - timedelta(days=1)

                # Format yesterday's date as "YYYYMMDD"
                yesterday_str = yesterday.strftime("%Y%m%d")

                # Replace "YYYYMMDD" in the pattern with yesterday's date
                input_file_pattern = source["input_file_pattern"].replace(
                    "YYYYMMDD", yesterday_str
                )
            elif date_condition == "TODAY":
                input_file_pattern = source["input_file_pattern"].replace(
                    "YYYYMMDD", date.today().strftime("%Y%m%d")
                )

            logger.info(f"Prepare sftp for {input_file_pattern}")

            source_file = source.get("sftp_source_folder")  ##+ '/' + input_file_pattern
            destination_file = source.get(
                "sftp_destination_folder"
            )  ##+ '/' + input_file_pattern

            sftp_config = source["sftp_source"]
            source_sftp = SFTPManager(
                sftp_config["host"],
                sftp_config["port"],
                sftp_config["username"],
                sftp_config.get("password", None),
            )
            source_dest_mode = source.get("source_dest_mode")

            if source_dest_mode == "different":
                dest_config = source.get("sftp_destination", None)
                # Copy a file to a different server
                dest_sftp = SFTPManager(
                    dest_config["host"],
                    dest_config["port"],
                    dest_config["username"],
                    dest_config.get("password", None),
                )
                try:
                    ##source_sftp.copy_file(source_file, destination_file, dest_sftp=dest_sftp)
                    source_sftp.copy_files(
                        source_file,
                        input_file_pattern,
                        destination_file,
                        dest_sftp=dest_sftp,
                    )
                    logger.info("File copied to different server successfully")
                finally:
                    dest_sftp.close()
            else:
                # Copy a file on the same server
                ##source_sftp.copy_file(source_file, destination_file)
                source_sftp.copy_files(
                    source_file, input_file_pattern, destination_file
                )
                logger.info("File copied to same server successfully")

            if source_archive_folder is not None:
                files_to_archive = source_sftp.sftp.listdir(source_file)
                for filename in files_to_archive:
                    if fnmatch.fnmatch(filename, input_file_pattern):
                        source_sftp.move_to_archive(
                            source_file, filename, source_archive_folder
                        )
                logger.info("Source Files moved to archive folder successfully")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the source connection
        source_sftp.close()


if __name__ == "__main__":
    import sys

    service_name = "recengine_report"
    if len(sys.argv) != 2:
        logger.info("Not found service name input, Use default=recengine_report")
    else:
        service_name = sys.argv[1]
    main(service_name)
