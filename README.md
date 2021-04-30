# OTUServer
## Overview:
Tutorial implementation of a simple HTTP server using pure *socket* level that handles GET and HEAD requests with
extensible request processing architecture based on *multiprocessing.pool.ThreadPool* functionality.

**This web server is be able to:**

- Scale to multiple workers
- **Count of workers** is specified by command line argument **-w**
- Respond with appropriate **200**, **403**, or **404** to **GET** requests and **HEAD** requests
- Respond **405** to other requests
- Return files at any path relative to **DOCUMENT_ROOT**.
- Calling `/file.html`  should return content `%DOCUMENT_ROOT%/file.html`
- **DOCUMENT_ROOT** is specified by command line argument **-r**
- Return `index.html` as directory index
- Calling `/directory/` should return `%DOCUMENT_ROOT%/directory/index.html`
- Respond with following headers for successful **GET** requests: **Date**, **Server**, **Content-Length**, **Content-Type**, **Connection**
- Correct **Content‑Type** for: `.html`, `.css`, `.js`, `.jpg`, `.jpeg`, `.png`, `.gif`, `.swf`
- Understand spaces and `%XX` in filenames

## Requirements:

- Python 3.x

- requirements.txt
```
docopt==0.6.2
schema==0.7.4
python_magic==0.4.22
```

## Usage:

```
Usage: httpd.py [options]

Options:
    -h --help                   Show this help message and exit
    --version                   Show version and exit
    -s --host=HOST              HOST is valid httpd server host name [default: 0.0.0.0] None/'' means all available interfaces.
    -p --port=PORT              PORT is valid httpd server port number on HOST [default: 0] if 0 then system chooses free port on HOST.
    -r --docsroot=DOCSROOT      DOCSROOT is httpd server working root files directory; '.' - current directory, <empty> - os.path.dirname(__file__).
    -w --workers=WORKERS        Count of workers. if None then system uses all available cores on HOST.
    -b --backlog=BACKLOG        Specify number of unaccepted connections that system will allow before refusing new connections [default: 0]; If 0 or not specified, a default reasonable value is chosen.
    -t --timeout=TIMEOUT        Set client socket operation timeout [default: 15] seconds.
    -c --chunksize=CHUNKSIZE    Set request chunk size [default: 1024] bytes.
    -m --maxsize=MAXSIZE        Set request max size [default: 4096] bytes.
    -l --loglevel=LOGLEVEL      LOGLEVEL is d (for DEBUG level) | i (INFO) | e (ERROR) [default: i].
    -g --generate-index         Enable generation of directory index page for directory request if this key present.

Example:
$ python httpd.py -p 8800 -r ~/Public
```

## Installation:
```
$ git clone https://github.com/nj-eka/otuserver.git
$ cd otuserver
$ pip install -r requirements.txt
```

## AB tesing results
```
$ ./httpd.py -r tests/httptest -p 8800 -w 8 -t 8

$ ab -n 50000 -c 100 -r http://localhost:8800/dir2
This is ApacheBench, Version 2.3 <$Revision: 1843412 $>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, http://www.zeustech.net/
Licensed to The Apache Software Foundation, http://www.apache.org/

Benchmarking localhost (be patient)
Completed 5000 requests
Completed 10000 requests
Completed 15000 requests
Completed 20000 requests
Completed 25000 requests
Completed 30000 requests
Completed 35000 requests
Completed 40000 requests
Completed 45000 requests
Completed 50000 requests
Finished 50000 requests


Server Software:        OTUServer/devel
Server Hostname:        localhost
Server Port:            8800

Document Path:          /dir2
Document Length:        34 bytes

Concurrency Level:      100
Time taken for tests:   124.321 seconds
Complete requests:      50000
Failed requests:        258
   (Connect: 0, Receive: 82, Length: 94, Exceptions: 82)
Total transferred:      11231550 bytes
HTML transferred:       1697212 bytes
Requests per second:    402.19 [#/sec] (mean)
Time per request:       248.641 [ms] (mean)
Time per request:       2.486 [ms] (mean, across all concurrent requests)
Transfer rate:          88.23 [Kbytes/sec] received

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0   29 280.3      0   15789
Processing:     1  182 4617.6      2  123302
Waiting:        0    5 168.1      1   30378
Total:          1  211 4663.9      2  124316

Percentage of the requests served within a certain time (ms)
  50%      2
  66%      3
  75%      3
  80%      4
  90%      7
  95%     10
  98%   1005
  99%   1032
 100%  124316 (longest request)
```