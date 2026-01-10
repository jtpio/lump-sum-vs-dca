#!/usr/bin/env python3
"""Inject custom CSS into the Specta index.html after JupyterLite build."""

from pathlib import Path

from bs4 import BeautifulSoup

CUSTOM_CSS = """
/* Custom styles for wider article view */
.specta-cell-output {
    max-width: 900px !important;
}
"""


def inject_css(output_dir=None):
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "_output"
    else:
        output_dir = Path(output_dir)
    index_path = output_dir / "specta" / "index.html"

    if not index_path.exists():
        print(f"Error: {index_path} not found. Run jupyter lite build first.")
        return False

    soup = BeautifulSoup(index_path.read_text(), "html.parser")

    # Check if already injected
    for style in soup.find_all("style"):
        if "Custom styles for wider article view" in style.string or "":
            print("Custom CSS already injected.")
            return True

    # Find the first style tag in head and append our CSS
    head = soup.find("head")
    if not head:
        print("Error: No <head> tag found.")
        return False

    style_tag = soup.new_tag("style")
    style_tag.string = CUSTOM_CSS
    head.append(style_tag)

    index_path.write_text(str(soup))
    print(f"Injected custom CSS into {index_path}")
    return True


if __name__ == "__main__":
    import sys

    output_dir = sys.argv[1] if len(sys.argv) > 1 else None
    inject_css(output_dir)
