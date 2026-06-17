import argparse

from qrref.settings import SETTINGS
from qrref.qr import (
    generate_qr_code,
    show_qr_code,
    save_qr_code,
    copy_qr_code,
)


# Create the parser
parser = argparse.ArgumentParser(description="Create a QR code and save, show or copy it.")
parser.add_argument("content", nargs="?", help="What to encode in the QR code", default="https://youtu.be/NR1TdhpCfcY")

ec_levels = parser.add_argument_group("Error correction level").add_mutually_exclusive_group()
ec_levels.add_argument("-L", action="store_true", help="Low: 7 percent error correction")
ec_levels.add_argument("-M", action="store_true", help="Medium: 15 percent error correction (default)")
ec_levels.add_argument("-Q", action="store_true", help="Quartile: 25 percent error correction")
ec_levels.add_argument("-H", action="store_true", help="High: 30 percent error correction")

parser.add_argument(
    "-v",
    "--version",
    type=int,
    default=0,
    help=(
        "Which version (size) of QR code to use. By default, chooses the smallest one"
        " that the content fits in."
    ),
)

formats = parser.add_argument_group("How to save the QR code").add_mutually_exclusive_group()
formats.add_argument("--png", action="store_true", help="As PNG")
formats.add_argument("--show", action="store_true", help="Show without saving (default)")
formats.add_argument("--copy", action="store_true", help="Copy image to clipboard")

parser.add_argument("-f", "--filename", help="Filename to save with", default="qr_code")
parser.add_argument("-t", "--transparent", action="store_true", help="Use transparent background")
parser.add_argument("-u", "--utf-8" , action="store_true", help="Use UTF-8 instead of Latin-1 encoding")
parser.add_argument("--debug", action="store_true", help="Show debug output")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.utf_8:
        SETTINGS.byte_encoding = "utf-8"

    # Find the selected ec_level, otherwise use "M"
    for ec_level, flag in zip(("L", "M", "Q", "H"), (args.L, args.M, args.Q, args.H)):
        if flag:
            break
    else:
        ec_level = "M"

    # Find the selected format, otherwise use "show"
    for format, flag in zip(("png", "show", "copy"), (args.png, args.show, args.copy)):
        if flag:
            break
    else:
        format = "show"


    version = args.version if 1 <= args.version <= 40 else None
    symbol = generate_qr_code(args.content, ec_level, version, debug=args.debug)

    match format:
        case "show":
            show_qr_code(symbol)
        case "png":
            file_path = args.filename + "." + format
            save_qr_code(symbol, file_path, args.transparent)
            print(f"QR code successfully saved to {file_path}.")
        case "copy":
            copy_qr_code(symbol, args.transparent)
            print("QR code succesfully copied to clipboard.")
