def main() -> None:
    import asyncio

    from .listen import main as listen

    try:
        asyncio.run(listen())
    except KeyboardInterrupt:
        pass
