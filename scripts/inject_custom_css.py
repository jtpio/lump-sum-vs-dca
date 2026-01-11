#!/usr/bin/env python3
"""Inject custom CSS into the Specta index.html after JupyterLite build."""

from pathlib import Path

from bs4 import BeautifulSoup

CSS_IDENTIFIER = "Custom styles for wider article view"
CUSTOM_CSS = f"""
/* {CSS_IDENTIFIER} */
.specta-cell-output {{
    max-width: 900px !important;
}}

/* Hide the loading spinner */
#specta-loader-host {{
    display: none !important;
}}

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
        style_text = style.string or ""
        if CSS_IDENTIFIER in style_text:
            if style_text.strip() == CUSTOM_CSS.strip():
                print("Custom CSS already injected.")
                return True
            style.string = CUSTOM_CSS
            index_path.write_text(str(soup))
            print("Updated existing custom CSS block.")
            return True

    # Find the head tag
    head = soup.find("head")
    if not head:
        print("Error: No <head> tag found.")
        return False

    # Prepend our CSS as the first style tag so it loads first
    style_tag = soup.new_tag("style")
    style_tag.string = CUSTOM_CSS
    # Insert after meta tags but before other styles
    first_style = head.find("style")
    if first_style:
        first_style.insert_before(style_tag)
    else:
        head.append(style_tag)

    # Note: we intentionally avoid mutating <body class=...> here.
    # Keeping this script limited to CSS/JS injection reduces layout flashes.

    index_path.write_text(str(soup))
    print(f"Injected custom CSS into {index_path}")
    return True


if __name__ == "__main__":
    import sys

    output_dir = sys.argv[1] if len(sys.argv) > 1 else None
    inject_css(output_dir)
