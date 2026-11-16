"""Backward-compatible launcher.

The server now lives in the convex_opt_mcp package. This file keeps existing
client configs that run `python server.py` working. Once you've installed the
package (`pip install -e .`) and pointed your client at the `convex-opt-mcp`
command, you can delete it.
"""

from convex_opt_mcp.server import main

if __name__ == "__main__":
    main()
