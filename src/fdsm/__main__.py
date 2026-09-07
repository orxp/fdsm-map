"""Allow ``python -m fdsm`` to invoke the CLI."""

from .cli import main

raise SystemExit(main())

