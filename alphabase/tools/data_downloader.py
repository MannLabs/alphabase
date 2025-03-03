"""Download files from sharing links."""

import base64
import cgi
import os
import traceback
import warnings
import zipfile
from abc import ABC, abstractmethod
from typing import Optional
from urllib.request import urlopen, urlretrieve

try:
    import progressbar  # noqa: F401

    _HAS_PROGRESSBAR = True
except ModuleNotFoundError:
    warnings.warn(
        "Dependency 'progressbar' not installed. Download progress will not be displayed.",
        ImportWarning,
    )
    _HAS_PROGRESSBAR = False


class Progress:  # pragma: no cover
    """Class to report the download progress of a file to the console."""

    def __init__(self):
        self.pbar = None

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        """Report download progress to console.

        Parameters
        ----------
        block_num : int
            number of blocks downloaded

        block_size : int
            size of each block in bytes

        total_size : int
            total size of the file to be downloaded in bytes

        """
        if not _HAS_PROGRESSBAR:
            return

        if total_size < 0:
            # disable progress when the downloaded item is a directory
            return

        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)  # noqa: F821
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


class FileDownloader(ABC):
    """Abstract class for downloading files from sharing links."""

    def __init__(self, url: str, output_dir: str):
        """Initialize FileDownloader.

        Parameters
        ----------
        url : str
            sharing link to download file from

        output_dir : str
            directory to save downloaded file to

        """
        self._url = url
        self._output_dir = output_dir
        self._encoded_url = self._encode_url()

        self._file_name = self._get_filename()
        self._is_archive = self._file_name.endswith(".zip")

        self._output_path = os.path.join(output_dir, self._file_name)
        self._unzipped_output_path = (
            self._output_path.replace(".zip", "")
            if self._is_archive
            else self._output_path
        )

    @abstractmethod
    def _encode_url(self) -> None:
        pass

    def download(self, max_size_kb: Optional[int] = None) -> str:  # pragma: no cover
        """Download file from sharing link.

        Parameters
        ----------
        max_size_kb : if given, the file will be truncated to this size in kilobytes, and the last line will be removed.
            The file name will be appended with '.truncated{max_size_kb}'.
            Note: don't use with binary files, as they will get corrupted!

        Returns
        -------
        local path to downloaded file
        """
        if self._is_archive and max_size_kb is not None:
            raise ValueError(
                "Cannot download a truncated archive. Please remove the max_size_kb parameter."
            )

        if max_size_kb:
            self._output_path = self._unzipped_output_path = (
                f"{self._output_path}.truncated{max_size_kb}"
            )

        if not os.path.exists(self._unzipped_output_path):
            print(f"{self._unzipped_output_path} does not yet exist")
            os.makedirs(self._output_dir, exist_ok=True)
            self._download_file(max_size_kb)

            if self._is_archive:
                self._handle_archive()

            size_mb = self._get_size() / 1024**2
            print(
                f"{self._unzipped_output_path} successfully downloaded ({size_mb} MB)"
            )
            # TODO check if its a binary file and warn/raise?
        else:
            size_mb = self._get_size() / 1024**2
            print(f"{self._unzipped_output_path} already exists ({size_mb} MB)")

        return self._unzipped_output_path

    def _get_size(self) -> int:
        """Return the size in bytes of the downloaded file or folder."""
        size = -1
        try:
            if os.path.isdir(self._unzipped_output_path):
                size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(
                        self._unzipped_output_path
                    )
                    for filename in filenames
                )
            else:
                size = os.path.getsize(self._unzipped_output_path)
        except Exception as e:
            print(f"Could not get size of {self._unzipped_output_path}: {e}")
        return size

    def _get_filename(self) -> str:  # pragma: no cover
        """Get filename from url."""
        try:
            remotefile = urlopen(self._encoded_url)
        except Exception as e:
            print(f"Could not open {self._url} for reading filename: {e}")
            raise ValueError(f"Could not open {self._url} for reading filename") from e

        info = remotefile.info()["Content-Disposition"]
        value, params = cgi.parse_header(info)
        return params["filename"]

    def _download_file(
        self, max_size_kb: Optional[int] = None
    ) -> None:  # pragma: no cover
        """Download file from sharing link.

        Returns
        -------
        path : str
            local path to downloaded file

        """
        try:
            if max_size_kb:
                with urlopen(self._encoded_url) as response, open(
                    self._output_path, "wb"
                ) as out_file:
                    out_file.write(response.read(max_size_kb * 1024))
                print(f"Truncating file to max. {max_size_kb} bytes ..")

                # truncate file as last line is most likely incomplete
                self._truncate_file(self._output_path)

            else:
                path, _ = urlretrieve(self._encoded_url, self._output_path, Progress())

        except Exception as e:
            print(f"{e} {traceback.print_exc()}")
            raise ValueError(f"Could not download {self._file_name}: {e}") from e

    @staticmethod
    def _truncate_file(file_path: str) -> None:
        """Remove the last line from a file."""
        with open(file_path) as file:
            lines = file.readlines()
        if lines:
            with open(file_path, "w") as file:
                file.writelines(lines[:-1])
            print(f"Truncated {file_path} to {len(lines) - 1} lines")

    def _handle_archive(self) -> None:
        """Unpack archive and remove it."""
        with zipfile.ZipFile(self._output_path, "r") as zip_ref:
            zip_ref.extractall(self._output_dir)
        print(f"{self._file_name} successfully unzipped")
        os.remove(self._output_path)


class OnedriveDownloader(FileDownloader):
    """Class for downloading files from onedrive sharing links."""

    def _encode_url(self) -> str:  # pragma: no cover
        """Encode onedrive sharing link as url for downloading files."""
        b64_string = base64.urlsafe_b64encode(str.encode(self._url)).decode("utf-8")
        encoded_url = f'https://api.onedrive.com/v1.0/shares/u!{b64_string.replace("=", "")}/root/content'
        return encoded_url


class DataShareDownloader(FileDownloader):
    """Class for downloading files from datashare sharing links."""

    def _encode_url(self) -> str:  # pragma: no cover
        """Encode datashare sharing link as url for downloading files."""
        # this is the case if the url points to a folder
        if "/download?" not in self._url:
            return f"{self._url}/download"

        return self._url
