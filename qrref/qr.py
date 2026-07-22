import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
try:
    import pyperclipimg  # type: ignore[reportMissingImports]
    COPY_ENABLED = True
except ImportError:
    COPY_ENABLED = False

from qrref.custom_types import *
from qrref.data_analysis import (
    select_modes,
    select_version,
)
from qrref.data_encoding import (
    encode_mixed,
    to_data_bytestream,
)
from qrref.error_correction import (
    generate_error_correction_blocks,
    interleave_blocks,
    generate_format_bits,
    generate_version_bits,
)
from qrref.placement import (
    place_bytestream,
    insert_finder_patterns,
    insert_alignment_patterns,
    insert_timing_patterns,
    place_format_bits,
    place_version_bits,
    expand_quiet_region,
)
from qrref.masking import (
    apply_mask,
)


def generate_qr_code(
    content: str, ec_level: ErrorCorrectionLevel, version: int | None = None, debug=False
) -> Symbol:
    """Generate a QR code symbol from the content to encode and the error correction level."""

    if version is None:
        version, modes = select_version(content, ec_level)
    else:
        modes = select_modes(content, version)
    if debug:
        print(f"{version = }")
        print(f"{modes = }")

    bitstream = encode_mixed(modes, version)
    bytestream = to_data_bytestream(bitstream, version, ec_level)
    blockstream = generate_error_correction_blocks(bytestream, version, ec_level)
    final_bytestream = interleave_blocks(blockstream, version, ec_level)

    symbol = place_bytestream(final_bytestream, version)
    insert_timing_patterns(symbol)
    insert_finder_patterns(symbol)
    insert_alignment_patterns(symbol, version)

    symbol, mask_pattern_id = apply_mask(symbol, version)
    if debug:
        print(f"{mask_pattern_id = }")

    place_format_bits(symbol, generate_format_bits(ec_level, mask_pattern_id))
    if version >= 7:
        place_version_bits(symbol, generate_version_bits(version))

    symbol = expand_quiet_region(symbol)
    return symbol


def qr_to_matplotlib(symbol: Symbol, transparent: bool):
    """Create an image array and colormap that Matplotlib understands."""
    image = symbol.astype(np.float64)
    cmap = plt.get_cmap("Greys")
    if transparent:
        image[~symbol] = np.nan
        cmap.set_bad(alpha=0)
    return image, cmap

def qr_to_pillow(symbol: Symbol, transparent: bool, min_width: int) -> Image.Image:
    """Create a PIL image of the symbol."""
    luma = (~symbol).astype(np.uint8) * 255
    if transparent:
        image = Image.fromarray(np.stack((luma, 255 - luma), axis=-1), mode="LA")
    else:
        image = Image.fromarray(luma, mode="L")

    # Upscale by an integer factor to at least min_width x min_width px
    width = len(symbol) * (1 + min_width // len(symbol))
    image = image.resize((width, width), Image.Resampling.NEAREST)

    return image


def show_qr_code(symbol: Symbol, transparent: bool = False):
    plt.rcParams["toolbar"] = "None"
    fig, ax = plt.subplots(figsize=(5, 5), num="QR code")  # num becomes the window title
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)  # No margins
    ax.set_axis_off()

    image, cmap = qr_to_matplotlib(symbol, transparent)
    ax.imshow(image, cmap=cmap, vmin=0, vmax=1)
    plt.show()


def save_qr_code(symbol: Symbol, file_path: str, transparent: bool = False, min_width: int = 500):
    image = qr_to_pillow(symbol, transparent, min_width)
    image.save(file_path)


def copy_qr_code(symbol: Symbol, transparent: bool = False, min_width: int = 500):
    if not COPY_ENABLED:
        raise ValueError(
            "Copying QR codes is not enabled. Reinstall using `pip install qrref[copy]` to enable."
        )

    image = qr_to_pillow(symbol, transparent, min_width)
    pyperclipimg.copy(image)  # type: ignore[reportPossiblyUnboundVariable]
