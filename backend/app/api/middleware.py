"""Bound actual bytes, including chunked bodies and inaccurate Content-Length."""
from starlette.responses import JSONResponse


class BodyLimitMiddleware:
    def __init__(self, app, max_bytes: int = 1_048_576):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.max_bytes:
                return await JSONResponse({"detail": "request body too large"}, 413)(
                    scope, receive, send)
            chunks.append(message)
            if not message.get("more_body", False):
                break
        position = 0

        async def bounded_receive():
            nonlocal position
            if position < len(chunks):
                item = chunks[position]
                position += 1
                return item
            return await receive()

        await self.app(scope, bounded_receive, send)
