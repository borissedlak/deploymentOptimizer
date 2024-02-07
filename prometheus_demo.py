from prometheus_client import start_http_server, Summary, Gauge
import random
import time

# Create a metric to track time spent and requests made.
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
cpu = Gauge('cpu', 'Description of gauge')

# Decorate function with metric.
# @REQUEST_TIME.time()
# def process_request(t):
#     """A consumer function that takes some time."""
#     time.sleep(t)
#     cpu.set(10)

if __name__ == '__main__':
    # Start up the server to expose the metrics.
    start_http_server(8000)
    # Generate some requests.
    while True:
        time.sleep(random.random())
        # process_request(random.random())