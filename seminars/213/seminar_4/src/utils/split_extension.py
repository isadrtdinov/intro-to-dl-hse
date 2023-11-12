def split_extension(filename: str) -> tuple:
    """
    Split filename into base name and file extension.

    Args:
        filename (str): Filename to be split.

    Returns:
        str: Base name.
        str: File extension.

    """
    split_pos = len(filename) - filename[::-1].index(".") - 1
    return filename[:split_pos], filename[split_pos:]
