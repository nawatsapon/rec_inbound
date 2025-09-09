import logging
import paramiko
import tempfile
import shutil
import glob
import sys
import zipfile  # Required for unzipping; typically placed at the top of the file
import fnmatch
import stat
import os

# Configure logging to output to both file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/wsol/app/logs/new_auto_reconcile/sftp_manager.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class SFTPManager:
    """Manages SFTP connections and file operations."""

    def __init__(self, host: str, port: int, username: str, password: str = None):
        """Initialize SFTP connection with RSA key or password."""
        private_key_file = os.path.expanduser("/wsol/app/.ssh/id_rsa")
        ##private_key_file = os.path.expanduser('C:/My/Python-Apps/FVD/reconcile/id_rsa')
        pkey = paramiko.RSAKey.from_private_key_file(private_key_file)
        self.transport = paramiko.Transport((host, port))
        self.transport.connect(
            username=username, pkey=pkey if not password else None, password=password
        )
        self.sftp = paramiko.SFTPClient.from_transport(self.transport)

    def download_files(
        self,
        remote_folder: str,
        pattern: str,
        local_folder: str,
        zip_file: bool = False,
        unzip_password: str = None,
        archive_folder: str = None,
    ) -> list:
        """Download files from SFTP matching the given pattern."""
        try:
            files = self.sftp.listdir(remote_folder)
            matched = [f for f in files if glob.fnmatch.fnmatch(f, pattern)]
            if not matched:
                logger.error(f"No files matching {pattern} om {remote_folder}")
                ##raise FileNotFoundError(f"No files matching {pattern} in {remote_folder}")
            downloaded_files = []
            for f in matched:
                logger.info(f"SFTP get: {remote_folder}/{f}, {local_folder}/{f}")
                self.sftp.get(f"{remote_folder}/{f}", f"{local_folder}/{f}")
                # Only attempt to unzip if download succeeded
                if zip_file:
                    try:
                        zip_file_name = f"{local_folder}/{f}"
                        logger.info(f"start unzip {zip_file_name}")
                        with zipfile.ZipFile(zip_file_name, "r") as zip_ref:
                            # Get the list of files in the zip
                            all_files = zip_ref.namelist()

                            # Filter for .csv, .xlsx, and .xls files (case-insensitive)
                            extracted_files = [
                                file
                                for file in all_files
                                if file.lower().endswith((".csv", ".xlsx", ".xls"))
                            ]

                            if not extracted_files:
                                logger.warning(
                                    f"No .csv, .xlsx, or .xls files found in {zip_file_name}"
                                )
                                raise

                            # Extract each file
                            for file_to_extract in extracted_files:
                                extracted_path = zip_ref.extract(
                                    file_to_extract,
                                    local_folder,
                                    pwd=(
                                        unzip_password.encode("utf-8")
                                        if unzip_password
                                        else None
                                    ),
                                )
                                downloaded_files.append(file_to_extract)
                            logger.info(
                                f"Extracted {len(extracted_files)} files from {zip_file_name} to {local_folder}"
                            )

                        # After successful extraction, archive the zip file
                        if archive_folder:
                            self.move_to_archive(remote_folder, f, archive_folder)
                            logger.info(f"Archived zip file {f} to {archive_folder}")

                    except zipfile.BadZipFile:
                        logger.error(f"{zip_file_name} is not a valid zip file")
                    except RuntimeError as e:
                        if "password" in str(e).lower():
                            logger.error(f"Incorrect password for {zip_file_name}")
                        else:
                            logger.error(f"Error unzipping {zip_file_name}: {e}")
                    except Exception as e:
                        logger.error(f"Unexpected error unzipping {zip_file_name}: {e}")
                else:
                    downloaded_files.append(f)
            return downloaded_files
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

    def move_to_archive(self, source_folder: str, filename: str, archive_folder: str):
        """Moves a single file to the archive folder, overwriting if it exists."""
        src = f"{source_folder}/{filename}"
        dest = f"{archive_folder}/{filename}"
        try:
            try:
                # If file exists in archive, remove it.
                self.sftp.stat(dest)
                self.sftp.remove(dest)
                logger.info(f"Removed existing file in archive: {dest}")
            except FileNotFoundError:
                # This is good, means we can just move it.
                pass

            self.sftp.rename(src, dest)
            logger.info(f"SFTP archived {src} to {dest}")
        except Exception as e:
            logger.error(f"SFTP Archive for {filename} failed: {e}")
            raise

    def copy_file(
        self, source_path: str, dest_path: str, dest_sftp: "SFTPManager" = None
    ):
        """
        Copy a file from source SFTP to destination SFTP.

        Args:
            source_path (str): Path to the source file on the source SFTP server.
            dest_path (str): Path where the file should be copied on the destination SFTP server.
            dest_sftp (SFTPManager, optional): SFTPManager instance for the destination server.
                                            If None, the destination is on the same server.

        Raises:
            ValueError: If source and destination paths are identical on the same server.
            Exception: If the copy operation fails (e.g., file not found, permission denied).
        """
        try:
            if dest_sftp is None:
                dest_sftp = self

            if dest_sftp is self:
                # Same server: copy directly using SFTP file handles
                if source_path == dest_path:
                    raise ValueError("Source and destination paths are the same")
                # logger.info(f"Copying {source_path} to {dest_path} on the same server")
                # with self.sftp.open(source_path, 'rb') as src_file:
                #     with self.sftp.open(dest_path, 'wb') as dest_file:
                #         shutil.copyfileobj(src_file, dest_file)

                # Different servers: download to temp file and upload
                logger.info(f"Copying {source_path} to {dest_path} on a same server")
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                try:
                    self.sftp.get(source_path, tmp_path)
                    self.sftp.put(tmp_path, dest_path)
                finally:
                    os.remove(tmp_path)
            else:
                # Different servers: download to temp file and upload
                logger.info(
                    f"Copying {source_path} to {dest_path} on a different server"
                )
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                try:
                    self.sftp.get(source_path, tmp_path)
                    dest_sftp.sftp.put(tmp_path, dest_path)
                finally:
                    os.remove(tmp_path)
        except Exception as e:
            logger.error(f"Copy failed: {e}")
            raise

    def copy_files(
        self,
        source_dir: str,
        file_pattern: str,
        dest_dir: str,
        dest_sftp: "SFTPManager" = None,
    ):
        """
        Copy files matching the pattern from source directory to destination directory.

        Args:
            source_dir (str): Path to the source directory on the source SFTP server.
            file_pattern (str): Pattern to match files in the source directory (e.g., '*.txt').
            dest_dir (str): Path to the destination directory on the destination SFTP server.
            dest_sftp (SFTPManager, optional): SFTPManager instance for the destination server.
                                            If None, the destination is on the same server.

        Raises:
            ValueError: If source directory does not exist, is not a directory, or if source and
                        destination directories are the same on the same server.
            Exception: If the copy operation fails for any file (e.g., file not found, permission denied).
        """
        # Set dest_sftp to self if not provided
        if dest_sftp is None:
            dest_sftp = self

        # Prevent copying to the same directory on the same server (would overwrite files)
        if dest_sftp is self and source_dir == dest_dir:
            raise ValueError("Source and destination directories are the same")

        # Verify source_dir exists and is a directory
        try:
            source_stat = self.sftp.stat(source_dir)
            if not stat.S_ISDIR(source_stat.st_mode):
                raise ValueError(f"{source_dir} is not a directory")
        except FileNotFoundError:
            raise ValueError(f"Source directory {source_dir} does not exist")

        # Get list of files in source_dir with attributes
        attrs = self.sftp.listdir_attr(source_dir)
        # Filter for regular files matching the pattern
        matching_files = [
            attr.filename
            for attr in attrs
            if stat.S_ISREG(attr.st_mode)
            and fnmatch.fnmatch(attr.filename, file_pattern)
        ]

        # Copy each matching file
        for file in matching_files:
            source_path = f"{source_dir}/{file}"
            dest_path = f"{dest_dir}/{file}"
            try:
                if dest_sftp is self:
                    # Same server: copy directly using file handles
                    logger.info(
                        f"Copying {source_path} to {dest_path} on the same server"
                    )
                    with self.sftp.open(source_path, "rb") as src_file:
                        with self.sftp.open(dest_path, "wb") as dest_file:
                            shutil.copyfileobj(src_file, dest_file)
                else:
                    # Different servers: download to temp file and upload
                    logger.info(
                        f"Copying {source_path} to {dest_path} on a different server"
                    )
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                    try:
                        logger.info(f"Get {source_path} to {tmp_path}")
                        self.sftp.get(source_path, tmp_path)
                        logger.info(f"put {tmp_path} to {dest_path}")
                        dest_sftp.sftp.put(tmp_path, dest_path)
                    finally:
                        os.remove(tmp_path)
            except Exception as e:
                logger.error(f"Failed to copy {source_path} to {dest_path}: {e}")
                raise

    def close(self) -> None:
        """Close SFTP connection."""
        self.sftp.close()
        self.transport.close()
