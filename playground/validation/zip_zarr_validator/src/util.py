# https://en.wikipedia.org/wiki/ZIP_(file_format)#ZIP64

import os
import zipfile


def check_for_zip64_signature(filename):
    """
    Checks if the file contains the ZIP64 end of central directory locator signature.
    """

    try:
        with open(filename, 'rb') as f:
            end_rec = zipfile._EndRecData(f)
            sig = end_rec[0] if end_rec else None
            return sig == zipfile.stringEndArchive64 or sig == zipfile.stringEndArchive64Locator
    except IOError as e:
        print(f"Error reading file: {e}")
        return False


def check_for_zip64_signature_raw(filename):
    """
    Checks the raw bytes of a file for the ZIP64 end of central directory locator signature.
    """
    # Define the magic signature for the Zip64 end of central directory locator
    # (bytes for 'PK\x06\x07')
    ZIP64_EOCD_LOCATOR_SIGNATURE = zipfile.stringEndArchive64Locator
    EOCD_RECORD_SIZE = 22  # Minimum size

    try:
        with open(filename, 'rb') as f:
            # Seek to near the end of the file to find the standard EOCD first
            f.seek(0 - EOCD_RECORD_SIZE, os.SEEK_END)
            # A more robust implementation would search for the EOCD signature backwards

            # For simplicity, assume the EOCD is at the end and check the bytes preceding it
            # This is a simplification and might not work for files with comments

            # A better way is to find the standard EOCD first, then check if a ZIP64 locator precedes it

            # Simplified check: search for the locator signature within a reasonable range from the end
            file_size = f.tell()
            search_window = min(file_size, 1024)  # Check the last 1KB of the file
            f.seek(0 - search_window, os.SEEK_END)
            data = f.read(search_window)

            return ZIP64_EOCD_LOCATOR_SIGNATURE in data

    except IOError as e:
        print(f"Error reading file: {e}")
        return False
