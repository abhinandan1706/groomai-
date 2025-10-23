# Enhanced Gunicorn configuration for GroomAI production deployment

import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 5000)}"
backlog = 2048

# Worker processes - optimized for AI model
workers = min(4, multiprocessing.cpu_count())  # Limit to 4 for memory considerations
worker_class = "sync"
worker_connections = 1000
timeout = 300  # Increased for AI processing
keepalive = 2

# Restart workers after this many requests to prevent memory leaks
max_requests = 500  # Lower for AI model to manage memory
max_requests_jitter = 50

# Logging
accesslog = "logs/access.log"
errorlog = "logs/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "groomai-prod"

# Server mechanics
preload_app = True
daemon = False
pidfile = "groomai.pid"

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Performance tuning for AI model
worker_tmp_dir = None
max_requests_jitter = 50
preload_app = True

# Memory management
worker_rlimit_nofile = 1024
worker_rlimit_fsize = 2 * 1024 * 1024 * 1024  # 2GB

# Graceful handling
graceful_timeout = 120
user = None
group = None
