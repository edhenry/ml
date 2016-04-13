from kafka import KafkaConsumer
import avro.schema
import avro.io
import io

import threading, logging, time


consumer = KafkaConsumer('netflow_records', group_id='my_group', bootstrap_servers = ['analytics1.solutions.brocade.com:6667'])

schema = avro.schema.parse(open('netflow.avsc').read())

for msg in consumer:
    bytes_reader = io.BytesIO(msg.value)
    decoder = avro.io.BinaryDecoder(bytes_reader)
    reader = avro.io.DatumReader(schema)
    flow_record = reader.read(decoder)
    print(flow_record)