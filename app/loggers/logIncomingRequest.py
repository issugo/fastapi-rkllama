import time
from typing import List

from fastapi import Request
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from .logger import Logger

logger = Logger.get_logger(__name__)


class LogIncomingRequest(BaseHTTPMiddleware):
    def __get_request_handler(self, req: Request):
        # get controllers from request
        routes: List[APIRoute] = req.app.routes
        for route in routes:
            if route.path_regex.match(req.url.path) and req.method in route.methods:
                return (
                    route.endpoint.__name__
                    if hasattr(route.endpoint, "__name__")
                    else "fastapi_core"
                )

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        func_name = self.__get_request_handler(request)
        request.state.func_name = func_name

        logger.info(
            f"{request.method} - {request.url.path} - {request.client.host} - {func_name} - start"
        )
        start_time = time.time()

        response = await call_next(request)

        process_time = (time.time() - start_time) * 1000
        formatted_process_time = "{0:.2f}".format(process_time)
        logger.info(f"{func_name} - end in time (ms): {formatted_process_time}")
        return response
