import numpy as np
import datetime

UTC_REFERENCE = 631065600

timestamp = 981073860
timestamp_16 = 120

ts_value = int(timestamp/2**16) * 2**16 + timestamp_16

value = datetime.datetime.utcfromtimestamp(UTC_REFERENCE + ts_value)
print(value)