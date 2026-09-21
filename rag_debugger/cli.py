"""Command-line entry point for RAG Debugger v0.2."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _prepare_imports() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _configure_stdio() -> None:
    if sys.platform != "win32":
        return
    import codecs

    if hasattr(sys.stdout, "buffer"):
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")


def _load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _is_report(data) -> bool:
    return isinstance(data, dict) and "claims" in data and "pipeline" in data


def run_analyze(path: str, as_json: bool = False) -> str:
    from rag.diagnose import diagnose
    from rag.trace import Trace

    report = diagnose(Trace.load(path))
    if as_json:
        return json.dumps(report.to_dict(), indent=2) + "\n"
    return report.format()


def run_compare(before_path: str, after_path: str) -> str:
    from rag.diagnose import diagnose
    from rag.report import compare_reports
    from rag.trace import Trace

    def load(path: str):
        data = _load_json(path)
        if _is_report(data):
            return data
        return diagnose(Trace.from_dict(data)).to_dict()

    return compare_reports(load(before_path), load(after_path))


def _serve(host: str, port: int, reload: bool) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.")
        print("  Windows:  $env:OPENAI_API_KEY='your-key-here'")
        print("  Linux/Mac: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    try:
        from app import app  # noqa: F401
    except Exception as exc:
        print(f"ERROR: could not import app: {exc}")
        sys.exit(1)

    print("Starting RAG Debugger...")
    print(f"  API:  http://{host}:{port}")
    print("  Stop: Ctrl+C")

    import uvicorn

    uvicorn.run("app:app", host=host, port=port, reload=reload)


def main(argv=None) -> None:
    _configure_stdio()
    _prepare_imports()

    parser = argparse.ArgumentParser(
        description="RAG Debugger tells you why your RAG failed and what to test next."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    sub = parser.add_subparsers(dest="command")

    analyze = sub.add_parser("analyze", help="Diagnose one trace JSON file")
    analyze.add_argument("trace")
    analyze.add_argument("--json", action="store_true", help="Print the DiagnosisReport as JSON")

    compare = sub.add_parser("compare", help="Compare two traces or two diagnosis reports")
    compare.add_argument("before")
    compare.add_argument("after")

    serve = sub.add_parser("serve", help="Start the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "analyze":
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is not set.")
            sys.exit(1)
        print(run_analyze(args.trace, as_json=args.json), end="")
        return
    if args.command == "compare":
        before = _load_json(args.before)
        after = _load_json(args.after)
        if not (_is_report(before) and _is_report(after)) and not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is not set. Compare saved reports, or set the key to diagnose traces.")
            sys.exit(1)
        print(run_compare(args.before, args.after), end="")
        return

    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8000)
    reload = getattr(args, "reload", False)
    _serve(host, port, reload)


if __name__ == "__main__":
    main()
