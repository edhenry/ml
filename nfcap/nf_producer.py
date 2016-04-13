from kafka import KafkaProducer, KafkaClient

import threading, logging, time

import io
import avro.schema
import avro.io

producer = KafkaProducer(bootstrap_servers='analytics1.solutions.brocade.com:6667')

def encode(schema_file, data):
    raw_bytes = None

    try:
        schema = avro.schema.parse(open(schema_file).read())
        writer = avro.io.DatumWriter(schema)
        bytes_writer = io.BytesIO()
        encoder = avro.io.BinaryEncoder(bytes_writer)
        writer.write(data, encoder)
        raw_bytes = bytes_writer.getvalue()
        print("Message sent!")
    except Exception as ex:
        template = "An exception of type {0} has occured. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
    return raw_bytes


def send(topic, raw_bytes):
    try:
        producer.send(topic, raw_bytes)
    except:
        print("Error sending message to Broker")

topic = 'netflow_records'

flow_record = {
    "timeReceived": 1458731295.314687,
    "nfHost": "10.110.111.170",
    "nfSourceID": 1,
    "sysUpTime": 1582258000,
    "unixSeconds": 1458729510,
    "sequenceNumber": 3562729,
    "flowSetID": 25,
    "IN_BYTES": 90,
    "IN_PKTS": 1,
    "PROTOCOL": 17,
    "TOS": 0,
    "TCP_FLAGS": 0,
    "L4_SRC_PORT": 55688,
    "IPV4_SRC_ADDR": "10.252.134.6",
    "INPUT_SNMP": 500031014,
    "L4_DST_PORT": 53,
    "IPV4_DST_ADDR": "10.70.20.23",
    "OUTPUT_SNMP": 500011031,
    "LAST_SWITCHED": 1582306000,
    "FIRST_SWITCHED": 1582306000,
    "IPV6_SRC_ADDR": "404",
    "IPV6_DST_ADDR": "404",
    "ICMP_TYPE": 0,
    "DIRECTION": 0,
    "flowId": 218201,
    "postNATSourceIPv4Address": "404",
    "postNATDestinationIPv4Address": "404",
    "postNAPTSourceTransportAddress": "404",
    "postNAPTDestinationTransportAddress": "404",
    "firewallEvent": 1,
    "postNATSourceIPv6Address": "404",
    "postNATDestinationIPv6Address": "404",
    "privateEnterpriseNumber": 25461,
    "App-ID": "incomplete",
    "User-ID": "brocade\pramaiah"
}

raw_bytes = encode('netflow.avsc', flow_record)

if raw_bytes is not None:
    for i in range(10000):
        send(topic, raw_bytes)
        print("Message %s sent!" % i)