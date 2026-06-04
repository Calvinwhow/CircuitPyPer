import html
import os
import webbrowser
from collections.abc import Mapping


class HTMLViewerSelector:
    """
    Build a standalone HTML selector for named HTML viewer objects.
    """

    def __init__(self, html_views: Mapping, title: str = "Map Viewer"):
        self.html_views = dict(html_views)
        self.title = title

    def _view_to_html(self, view):
        """
        Convert a nilearn/IPython-style HTML object into an HTML string.
        """
        if hasattr(view, "_repr_html_"):
            return view._repr_html_()
        if hasattr(view, "get_standalone"):
            return view.get_standalone()
        if hasattr(view, "html"):
            value = view.html
            return value() if callable(value) else value
        return str(view)

    def to_html(self):
        """
        Return a standalone HTML document with a dropdown map selector.
        """
        if not self.html_views:
            raise ValueError("html_views must contain at least one named HTML object.")

        names = [str(name) for name in self.html_views.keys()]
        options = "\n".join(
            f'<option value="{i}">{html.escape(name)}</option>'
            for i, name in enumerate(names)
        )

        frames = []
        for i, name in enumerate(names):
            view_html = self._view_to_html(self.html_views[name])
            hidden = "" if i == 0 else " hidden"
            frames.append(
                '<iframe '
                f'id="viewer-{i}" '
                f'title="{html.escape(name)}" '
                f'srcdoc="{html.escape(view_html, quote=True)}"'
                f'{hidden}></iframe>'
            )

        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(self.title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f5f5f5;
      color: #222;
    }}
    header {{
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px 16px;
      background: #fff;
      border-bottom: 1px solid #ddd;
    }}
    label {{
      font-weight: 600;
    }}
    select {{
      min-width: 280px;
      max-width: 70vw;
      padding: 6px 8px;
      font-size: 14px;
    }}
    main {{
      height: calc(100vh - 57px);
    }}
    iframe {{
      width: 100%;
      height: 100%;
      border: 0;
      background: #fff;
    }}
    iframe[hidden] {{
      display: none;
    }}
  </style>
</head>
<body>
  <header>
    <label for="viewer-select">Map</label>
    <select id="viewer-select">{options}</select>
  </header>
  <main>
    {"".join(frames)}
  </main>
  <script>
    const selector = document.getElementById("viewer-select");
    const frames = Array.from(document.querySelectorAll("iframe"));
    selector.addEventListener("change", () => {{
      frames.forEach((frame, index) => {{
        frame.hidden = String(index) !== selector.value;
      }});
    }});
  </script>
</body>
</html>
"""

    def save(self, out_file):
        """
        Write the selector HTML document to disk and return the output path.
        """
        out_dir = os.path.dirname(out_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(self.to_html())
        return out_file

    def open_in_browser(self, out_file):
        """
        Write the selector HTML document to disk, open it in a browser, and return the output path.
        """
        out_file = self.save(out_file)
        webbrowser.open(f"file://{os.path.abspath(out_file)}")
        return out_file
