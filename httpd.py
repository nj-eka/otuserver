#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""httpd: simple http server

Usage: httpd.py [options]

Options:
    -h --help                   Show this help message and exit
    --version                   Show version and exit
    -s --host=HOST              HOST is valid httpd server host name [default: 0.0.0.0] None/'' means all available interfaces.
    -p --port=PORT              PORT is valid httpd server port number on HOST [default: 0] if 0 then system chooses free port on HOST.
    -r --docsroot=DOCSROOT      DOCSROOT is httpd server working root files directory; '.' - current directory, <empty> - os.path.dirname(__file__).
    -w --workers=WORKERS        Count of workers. if None then system uses all available cores on HOST.
    -b --backlog=BACKLOG        Specify number of unaccepted connections that system will allow before refusing new connections [default: 0]; If 0 or not specified, a default reasonable value is chosen.
    -t --timeout=TIMEOUT        Set client socket operation timeout [default: 60] seconds.
    -c --chunksize=CHUNKSIZE    Set request chunk size [default: 1024] bytes.
    -m --maxsize=MAXSIZE        Set request max size [default: 4096] bytes.
    -l --loglevel=LOGLEVEL      LOGLEVEL is d (for DEBUG level) | i (INFO) | e (ERROR) [default: i].
    -g --generate-index         Enable generation of directory index page for directory request if this key present.
"""

import os
import io
import sys
import logging
import typing
import time
import socket
from pathlib import Path
from urllib.parse import urlsplit, quote, unquote
from html import escape
import mimetypes
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from enum import IntEnum
from contextlib import suppress, closing

from docopt import docopt, DocoptExit  # https://pypi.org/project/docopt/
from schema import Schema, And, Or, Use, Optional, SchemaError  # https://github.com/keleshev/schema https://semver.org/
from chardet.universaldetector import UniversalDetector
import magic

__version__ = "devel"

SERVER_VERSION = f"SimpleFileHTTPRequestServer/{__version__}"
PROTOCOL_TYPE_SUPPORTED = ["HTTP"]  # todo: add HTTPS support
PROTOCOL_VERSIONS_SUPPORTED = [(0, 9), (1, 0), (1, 1)]
# https://docs.python.org/3/library/http.server.html#http.server.BaseHTTPRequestHandler
# The default request version.  This only affects responses up until
# the point where the request line is parsed, so it mainly decides what
# the client gets back when sending a malformed request line.
# Most web servers default to HTTP 0.9, i.e. don't send a status line.
DEFAULT_REQUEST_VERSION = "HTTP/0.9"

# https://www.w3.org/International/articles/http-charset/index.ru
# https://developer.mozilla.org/ru/docs/Web/HTTP/Headers/Accept-Charset
HTTP_DEFAULT_ENCODING = 'iso-8859-1'    # 'utf-8'
INDEX_FILE_NAME = 'index.html'
END_LINE = "\r\n"
HTTP_DATE_FORMAT = '%a, %d %b %Y %H:%M:%S GMT'
SERVER_NAME = 'OTUServer'
H_SIZE_ABBREVIATIONS = ['bytes', 'KB', 'MB', 'GB', 'TB']

class HTTPStatus(IntEnum):  # https://httpstatuses.com/
    """Shorthand version of original class from http module for server code supported response."""
    def __new__(cls, value, phrase, description=''):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.phrase = phrase
        obj.description = description
        return obj

    def __repr__(self):
        return "{!r}({!r},{!r},{!r})".format(self.__class__, self.value, self.phrase, self.description)  # pylint: disable=maybe-no-member

    def __str__(self):
        return f'{self.value} {self.phrase}\n{self.description}'  # pylint: disable=maybe-no-member

    # CONTINUE = 100, 'Continue', 'Request received, please continue'

    # success
    OK = (200, 'OK', 'Request fulfilled, document follows')

    # client error
    BAD_REQUEST = (400, 'Bad Request', 'Bad request syntax or unsupported method')
    FORBIDDEN = (403, 'Forbidden', 'Request forbidden -- authorization will not help')
    NOT_FOUND = (404, 'Not Found', 'Nothing matches the given URI')
    METHOD_NOT_ALLOWED = (405, 'Method Not Allowed', 'Specified method is invalid for this resource')
    NOT_ACCEPTABLE = (406, 'Not Acceptable', 'URI not available in preferred format')
    REQUEST_TIMEOUT = (408, 'Request Timeout', 'Request timed out; try again later')
    REQUEST_ENTITY_TOO_LARGE = (413, 'Request Entity Too Large', 'Entity is too large')
    REQUEST_URI_TOO_LONG = (414, 'Request-URI Too Long', 'URI is too long')
    UNSUPPORTED_MEDIA_TYPE = (415, 'Unsupported Media Type', 'Entity body in unsupported format')
    REQUEST_HEADER_FIELDS_TOO_LARGE = (431, 'Request Header Fields Too Large',
                                            'The server is unwilling to process the request because its header '
                                            'fields are too large')

    # server errors
    INTERNAL_SERVER_ERROR = (500, 'Internal Server Error', 'Server got itself in trouble')
    NOT_IMPLEMENTED = (501, 'Not Implemented', 'Server does not support this operation')
    HTTP_VERSION_NOT_SUPPORTED = (505, 'HTTP Version Not Supported', 'Cannot fulfill request')


class HTTPException(Exception):
    """Exception class for passing http status with some details."""
    def __init__(self, status: HTTPStatus = HTTPStatus.INTERNAL_SERVER_ERROR, details: str = None, exception: Exception = None):
        super().__init__()
        self.status: HTTPStatus = status
        self.details = details if details else exception
        if not self.details and sys.exc_info()[1]:
            self.details = f'{sys.exc_info()[1]}'

    def __str__(self):
        return f'{str(self.status)}' + (f'\ndetails: {self.details}' if self.details else '')

class Protocol:
    """Protocol string validated parts"""
    def __init__(self, protocol_string: str):
        protocol_type, _, protocol_version = protocol_string.upper().partition("/")
        if protocol_type not in PROTOCOL_TYPE_SUPPORTED:
            raise HTTPException(HTTPStatus.BAD_REQUEST, f'Unacceptable protocol name: {protocol_type}')
        if not protocol_version:
            raise HTTPException(HTTPStatus.BAD_REQUEST, f'Invalid protocol version: {protocol_string}')
        protocol_version_number = tuple(map(int, protocol_version.split(".")))
        if protocol_version_number not in PROTOCOL_VERSIONS_SUPPORTED:
            raise HTTPException(HTTPStatus.BAD_REQUEST, f'Unsupported HTTP version: {protocol_version_number}')
        self.type: str = protocol_type
        self.version: tuple = protocol_version_number

    def __str__(self):
        return f'{self.type}/{".".join(map(str,self.version))}'


class HTTPRequest(typing.NamedTuple):
    """Parsed and validated http request class."""
    protocol: Protocol
    command: str
    rpath: str
    abs_path: Path
    docs_root: Path
    headers: OrderedDict
    body: str
    timeout: float
    ctime: float

    @classmethod
    def parse_request(cls, request_raw: bytes, docs_root: Path, generate_index: bool = False, timeout: float = None) -> typing.NamedTuple:
        """Parse and validate http raw request."""
        if not request_raw:
            raise HTTPException(HTTPStatus.BAD_REQUEST, 'Empty request')

        lines = request_raw.decode(HTTP_DEFAULT_ENCODING).rstrip(END_LINE).split(END_LINE)

        request_line_parts = lines[0].split()
        if len(request_line_parts) not in [2, 3]:
            raise HTTPException(HTTPStatus.BAD_REQUEST, f'Invalid request line format: {lines[0]}')

        command, uri = request_line_parts[:2]

        if len(request_line_parts) == 2 and command != 'GET':
            raise HTTPException(HTTPStatus.BAD_REQUEST, f'Bad HTTP/0.9 request type: {command}')

        protocol = Protocol(DEFAULT_REQUEST_VERSION if len(request_line_parts) == 2 else request_line_parts[-1])

        rpath = unquote(urlsplit(uri).path).lstrip('/')
        logging.debug('%s wanted.', rpath)
        if any(part in rpath.split('/') for part in ['~', '.', '..']):
            raise HTTPException(HTTPStatus.FORBIDDEN, f'Forbidden path format: {rpath}')
        abs_path = docs_root.joinpath(rpath)
        if abs_path.is_file() and rpath.endswith('/'):
            raise HTTPException(HTTPStatus.NOT_FOUND)
        if abs_path.is_dir() and not generate_index:
            abs_path = abs_path.joinpath(INDEX_FILE_NAME)
        if not abs_path.exists():
            raise HTTPException(HTTPStatus.NOT_FOUND)

        headers = OrderedDict(list((key.title(), value.strip().lower()) for key, _, value in (line.partition(':') for line in lines[1:] if line)))
        return cls(
            protocol=protocol,
            command=command,
            rpath=rpath,
            abs_path=abs_path,
            docs_root=docs_root,
            headers=headers,
            timeout=timeout,
            ctime=time.time(),
            body=None)


def gmtime_string(timestamp: int = None, format_time: str = HTTP_DATE_FORMAT):
    """Return [timestamp] as [format_time] GMT string. If [timestamp] is None then use current time."""
    return time.strftime(format_time, time.gmtime(timestamp))  # == datetime.datetime.utcfromtimestamp(timestamp).strftime(HTTP_DATE_FORMAT) & datetime.datetime.utcnow().strftime(HTTP_DATE_FORMAT)

def h_size(size: int, units: list = None):
    """ Returns a human readable string representation of bytes """
    if units is None:
        units = H_SIZE_ABBREVIATIONS
    return f'{size} {units[0]}' if size < 1024 else h_size(size >> 10, units[1:])

def mime_magic_string(file_path: Path, mime_type: bool=False, mime_encoding: bool=False, uncompress: bool=False) -> str:
    """ Return magic mime string of file."""
    f_m = magic.Magic(mime=mime_type, mime_encoding=mime_encoding, uncompress=uncompress)
    return f_m.from_file(str(file_path))

# https://www.w3.org/International/articles/http-charset/index.ru
def mime_string(file_path: Path, text_signature_detection: bool = False) -> str:
    """ Return mime string of file."""
    guess, encoding = mimetypes.guess_type(file_path.resolve())  # symbolic links should be resolved for mimetypes.guess_type(...) to work correctly
    guess = guess if guess else 'application/octet-stream'
    if guess.startswith('text/') and not encoding and text_signature_detection:
        with suppress(Exception):
            with closing(UniversalDetector()) as enc_detector, open(file_path, 'rb') as file:
                for line in file:
                    enc_detector.feed(line)
                    if enc_detector.done:
                        encoding = enc_detector.result['encoding']
                        break
    if encoding:
        return f'{guess};charset={encoding}'
    return guess


# https://developer.mozilla.org/en-US/docs/Web/HTTP/Messages
class HTTPResponse:
    """Base class that implements basic functionality for setting up and sending response."""
    def __init__(self, request: HTTPRequest, status: HTTPStatus, headers: list = None, default_response_version: str = 'HTTP/1.0'):
        self.request: HTTPRequest = request  #  can be None for ErrorHTTPResponse if parsing failed
        self.protocol: Protocol = request.protocol if request else Protocol(default_response_version)
        self.status: HTTPStatus = status

        self.__headers = OrderedDict()
        self.set_header('Server', self.get_server_name())
        self.set_header('Date', gmtime_string())
        if headers:
            for header in headers:
                if isinstance(header, str):
                    key, _, value = header.partition(':')
                elif isinstance(header, (list, tuple)):  # isinstance(header, collections.abc.Sequence)
                    key, _, value = header[0], ':', '; '.join(map(str.strip, header[1:]))
                self.set_header(key, value)
        self._content_io = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.request = None
        self.status = None
        self.__headers = None
        if self._content_io:
            self._content_io.close()

    def get_status_line(self):
        """Return status line string."""
        return f'{str(self.protocol)} {int(self.status)} {self.status.phrase}'

    def get_headers(self):
        """Return list of header string as 'key: value'."""
        return [f'{key}: {value}' for key, value in self.__headers.items()]

    def get_header(self, key: str, default_value: object = '') -> object:
        """Return header value."""
        return self.__headers.get(key.title(), default_value).lower()

    def close_connection(self) -> bool:
        """Return True if value of 'Connection' header is not 'keep-alive'."""
        return self.get_header('Connection') != 'keep-alive'

    def set_header(self, name: str, value: str) -> None:
        """Set header value in correct format."""
        self.__headers[name.title()] = value  # value.lower()

    def set_text_content(self, text: str, encoding: str = sys.getdefaultencoding(), encoding_errors: str = "strict", text_type: str = "text/html"):
        """Set body and content headers from text."""
        content_encoded = text.encode(encoding, encoding_errors)
        self.set_header("Content-Type", f"{text_type}; charset={encoding}")
        self.set_header("Content-Length", str(len(content_encoded)))
        self.set_header("Last-Modified", gmtime_string())
        self._content_io = io.BytesIO()
        self._content_io.write(content_encoded)
        self._content_io.seek(0)

    def set_file_content(self, file_path: Path, content_type: str):
        """Set body and content headers from file."""
        file_stat = file_path.stat()
        self.set_header("Content-Length", str(file_stat.st_size))
        self.set_header("Last-Modified", gmtime_string(file_stat.st_mtime))
        self.set_header("Content-Type", content_type)
        self._content_io = io.open(file_path, 'rb')

    def send_head(self, conn: socket.socket, encoding: str = HTTP_DEFAULT_ENCODING) -> bool:
        """Send status line and headers into [conn]."""
        conn_addr = conn.getpeername()
        if 'Connection' not in self.__headers:
            self.set_header('Connection', self.request.headers.get('Connection', 'close').lower() if self.request else 'close')
        if self.get_header('Connection') == 'keep-alive' and (timeout := conn.gettimeout()):
            logging.debug('%s timeout = %d', conn_addr, timeout)
            self.set_header('Keep-Alive', f"timeout={int(timeout + self.request.ctime - time.time() if self.request else timeout)}")
        head = END_LINE.join([self.get_status_line(), *self.get_headers(), END_LINE]).encode(encoding)
        logging.debug('%s <- sending head [%s]', conn_addr, head)
        if self.protocol.version != (0, 9):
            with suppress(BrokenPipeError):
                conn.sendall(head)
                logging.info('%s <- head sent - ok', conn_addr)

    def send_body(self, conn: socket.socket):
        """Send body into [conn]."""
        if self._content_io and (not self.request or self.request.command != "HEAD"):
            logging.debug('%s <- sending body', conn.getpeername())
            with suppress(BrokenPipeError):
                conn.sendfile(self._content_io)
                logging.info('%s <- body sent - ok', conn.getpeername())

    def send_all(self, conn: socket.socket, encoding: str = HTTP_DEFAULT_ENCODING):
        """Send head and body into conn."""
        self.send_head(conn, encoding)
        self.send_body(conn)

    @staticmethod
    def get_server_name():
        """Return server name."""
        return f'{SERVER_NAME}/{__version__}'


class ErrorHTTPResponse(HTTPResponse):
    """Class for error response sending to client."""
    content_encoding = 'utf-8'
    content_type = "text/html"
    content_template = """\
    <!DOCTYPE html>
    <html>
        <head>
            <meta http-equiv="Content-Type" content="text/html; charset={encoding}">
            <title>Error response</title>
        </head>
        <body>
            <h1>Error response:</h1>
            <p>Error code: {code}</p>
            <p>Message: {phrase}</p>
            <p>Details: {details}</p>
        </body>
    </html>
    """

    def __init__(self, request: HTTPRequest, status: HTTPStatus, details: str = None, headers: list = None):
        super().__init__(request, status, headers)
        self.set_header('Connection', 'close')
        content = self.content_template.format(
                        encoding=self.content_encoding,
                        code=int(status),
                        phrase=escape(status.phrase, quote=False),
                        details=escape(details if details else status.description, quote=False),
                    )
        self.set_text_content(content, self.content_encoding, 'replace', self.content_type)


# todo: use https://noamkremen.github.io/a-simple-threadsafe-caching-decorator.html
class FileHTTPResponse(HTTPResponse):
    """Class for requested file content sending to client."""
    index_template = '<!DOCTYPE html>'\
                        '<html><head><meta http-equiv="Content-Type" content="text/html; charset={encoding}"><title>{title}</title></head>'\
                        '<body><h1>{title}</h1><hr>'\
                        '<ul>{list_index}</ul>'\
                        '<hr></body></html>'
    title_template = 'Index of {link}'
    list_index_template = '<li>{link} <i>{mtime}</i> <b>{size}</b> {mime}</li>'
    link_template = '<a href="{href}">{name}</a>'

    def __init__(self, request, headers: list = None):
        super().__init__(request, HTTPStatus.OK, headers)
        if request.abs_path.is_dir():
            self.set_text_content(self.directory_index_html(), encoding_errors='surrogateescape')
        else:
            self.set_file_content(request.abs_path, mime_string(request.abs_path, text_signature_detection=True))

    @classmethod
    def link_html(cls, ref: str, name: str):
        """Return formated <a href...> string."""
        return cls.link_template.format(
            href=quote(("/" if not ref.startswith('/') else '') + ref, errors='surrogatepass'),
            name=escape(name, quote=False)
            )

    def directory_index_html(self, encoding: str = sys.getfilesystemencoding()) -> str:
        """Generate index list directory."""
        li_rows = []
        rel_path = Path(self.request.rpath)
        dir_list = [('..', True, False, rel_path.parent, '', '', '')] if self.request.rpath else []
        try:
            dir_list += sorted([(
                            x.name,
                            x.is_dir(),
                            x.is_symlink(),
                            rel_path.joinpath(x.name),
                            time.strftime('%Y-%m-%d %H:%M:%S GMT', time.gmtime(x.stat().st_mtime)),
                            h_size(x.stat().st_size) if x.is_file() else '',
                            mime_magic_string(x, mime_type=False, mime_encoding=False, uncompress=True) if x.is_file() else '')
                            for x in self.request.abs_path.iterdir()],
                        key=lambda x: (-x[1], x[0].lower()))
        except OSError as os_exc:
            raise HTTPException(HTTPStatus.FORBIDDEN, f'No permission to list directory: {self.request.rpath}') from os_exc
        for f_name, is_dir, is_sl, f_path, f_mtime, f_size, f_mime in dir_list:
            display_name = f_name + ('/' if is_dir else '') + ('@' if is_sl else '')
            li_rows.append(self.list_index_template.format(
                link = self.link_html(str(f_path), display_name),
                mtime=f_mtime,
                size=f_size,
                mime= f_mime
            ))
        rel_parents = [(p.name, f'{str(p)}') for p in [rel_path, *rel_path.parents]][:-1] + [('_', '')]
        rel_parents.reverse()
        title = self.title_template.format(link = '>'.join([self.link_html(ref = rp_path, name = rp_name) for rp_name, rp_path in rel_parents]))
        return self.index_template.format(
            encoding=encoding,
            title=title,
            list_index=''.join(li_rows),
        )


class FileHTTPRequestHandler:
    """This class is used to handle the HTTP requests that arrive at server."""
    END_REQUEST_RAW = (END_LINE*2).encode(HTTP_DEFAULT_ENCODING)
    def __init__(self, client_socket: socket.socket, client_address: int, docs_root: Path, request_chunk_size: int, request_max_size: int, client_socket_timeout: float = None, generate_index: bool = False):
        self.conn = client_socket
        self.conn.settimeout(client_socket_timeout)
        self.conn_timeout = client_socket_timeout
        self.conn_addr = client_address
        self.docs_root = docs_root
        self.generate_index = generate_index
        self.request_chunk_size = request_chunk_size
        self.request_max_size = request_max_size
        self.request = None
        try:
            self.start()
        finally:
            self.finish()

    def start(self):
        """Start processing requests."""
        logging.info('%s - start processing requests.', self.conn_addr)
        while not self.handle_request_complete():
            pass

    def finish(self):
        """Finish processing request."""
        self.conn.close()
        logging.info('%s - stop processing requests.', self.conn_addr)

    def handle_request_complete(self) -> bool:
        """Handle a single HTTP request."""
        try:
            request_raw = self.read_socket(self.conn, self.request_chunk_size, self.request_max_size)
            logging.debug('%s -> %s', self.conn_addr, request_raw[:100])
            # If conn.recv() returns an empty bytes object, b'', then the client closed the connection and the loop is terminated.
            if not request_raw:
                logging.info('%s - closed by client.', self.conn_addr)
                return True
            self.request = HTTPRequest.parse_request(request_raw, self.docs_root, self.generate_index, self.conn_timeout)
            logging.info('%s -> %s %s %s', self.conn_addr, self.request.protocol, self.request.command, self.request.rpath)
            do_command = 'do_' + self.request.command.lower()
            if hasattr(self, do_command):
                return getattr(self, do_command)()  # todo: add support for Expect directive https://developer.mozilla.org/ru/docs/Web/HTTP/Status/100
            else:
                return self.send_error(HTTPStatus.METHOD_NOT_ALLOWED, f'Method {self.request.command} not allowed.', [('Allow', 'GET, HEAD')])
        except HTTPException as exc:
            return self.send_error(exc.status, exc.details)
        except socket.timeout:
            return self.send_error(HTTPStatus.REQUEST_TIMEOUT)
        except BaseException as exc:  # pylint: disable=broad-except
            return self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, repr(exc))
        return True

    def do_get(self) -> bool:
        """Serve a GET request."""
        with FileHTTPResponse(self.request) as response:
            response.send_all(self.conn)
            return response.close_connection()

    def do_head(self) -> bool:
        """Serve a HEAD request."""
        with FileHTTPResponse(self.request) as response:
            response.send_head(self.conn)
            return response.close_connection()

    def send_error(self, status: HTTPStatus, details: str = None, headers: list = None) -> bool:
        """Log error and try to send error message to client."""
        logging.error('%s -> %d %s', self.conn_addr, status, details )
        with suppress(BaseException),  ErrorHTTPResponse(self.request, status, details, headers) as response:
            response.send_all(self.conn)
        return True

    @classmethod
    def read_socket(cls, conn: socket.socket, request_chunk_size: int, request_max_size: int) -> bytes:
        """ Read raw data from socket by chunks."""
        data = b''
        while True:
            if (chunk := conn.recv(request_chunk_size)):
                data += chunk
                if len(data) > request_max_size:
                    raise HTTPException(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f'Request exceeded maximum length: {request_max_size}')
                if cls.END_REQUEST_RAW in data:
                    break
            else:
                break
        return data


class OTUServer:
    """
    Simple HTTP server using pure socket level that handles GET and HEAD requests
    with extensible request processing architecture based on multiprocessing.pool.ThreadPool functionality.
    """
    def __init__(self, host: str, port: int, workers: int, backlog: int, handler=FileHTTPRequestHandler,  **kwargs) -> None:
        self.host: str = host
        self.port: int = port
        self.workers: int = workers
        self.backlog: int = backlog
        self.handler: FileHTTPRequestHandler = handler
        self.handler_params: dict = kwargs

    def serve_forever(self):
        """Serve forever and ever."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(self.backlog)
            logging.info('Start listening on %s', server_socket.getsockname())
            # https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                while True:
                    try:
                        client_socket, client_address = server_socket.accept()
                        logging.info('Request from %s is accepted', client_socket.getpeername())  # client_socket.getpeername() == client_address
                        executor.submit(self.handler, client_socket, client_address, **self.handler_params)
                    except KeyboardInterrupt:
                        executor.shutdown(wait=True)
                        logging.info('Server %s stopped', server_socket.getsockname())
                        break


if __name__ == "__main__":
    try:
        args = docopt(__doc__, version=__version__)
        schema = Schema({
            Optional('--loglevel'): And(str, Use(lambda l: {'d': logging.DEBUG, 'i': logging.INFO, 'e': logging.ERROR}[l.lower()]), error='Unexpected --loglevel value.'),  # Regex(r'^[eid]$', flags=re.I),
            Optional('--docsroot'): And(Use(lambda p: p if p else os.path.dirname(__file__)), os.path.exists, Use(lambda r: Path(r).expanduser().resolve(strict=True)), error='DOCSROOT doesn"t exist.'),
            Optional('--workers'): Or(None, And(Use(int), lambda n: n >= 0, error='if specified, WORKERS should be integer >=0')),
            Optional('--host'): Use(socket.gethostbyname, error="Invalid hostname."),
            Optional('--port'): And(Use(int), lambda n: 0 <= n < 65536, error='Port numver should be integer >=0 and < 65536'),
            Optional('--backlog'): Use(int),
            Optional('--timeout'): Or(None, Use(float)),
            Optional('--chunksize'): Use(int),
            Optional('--maxsize'): Use(int),
            Optional('--generate-index'): Use(bool),
            Optional('--version'): Use(bool),
            Optional('--help'): Use(bool)})
        args = schema.validate(args)
        logging.basicConfig(
            format='[%(asctime)s]%(relativeCreated)6d %(levelname).1s %(processName)s.%(thread)d-%(threadName)s\n%(message)s',
            datefmt='%Y.%m.%d %H:%M:%S',
            level=args['--loglevel'],
            force=True
        )
        logging.debug('Validated params: %s', args)
        server = OTUServer(
            host=args['--host'],
            port=args['--port'],
            docs_root=args['--docsroot'],
            workers=args['--workers'],
            backlog=args['--backlog'],
            client_socket_timeout=args['--timeout'],
            request_chunk_size=args['--chunksize'],
            request_max_size=args['--maxsize'],
            generate_index=args['--generate-index'],
            )
        server.serve_forever()
    except DocoptExit as exc:
        logging.error('Invalid keywords usage %s\n%s', exc, __doc__)
    except SchemaError as exc:
        logging.error('Invalid keyword argument value.\nError: %s', exc)
    # do not use bare 'except' - pycodestyle(E722)
    except BaseException:   # pylint: disable=broad-except
        logging.exception("Oops...", exc_info=True)
