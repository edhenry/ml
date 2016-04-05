from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
from vispy import app
from vispy import gloo

np.random.seed(1234)
np.set_printoptions(threshold='nan')

# Import netflow capture file(s)

# Crate dataframe
brocade_flowdata = pd.DataFrame()

# List of csv's to read in
brocade_cap_files = ["/home/ehenry/code/data/1458576494.csv"]

# Read in the csv's and append to dataframe
for f in brocade_cap_files:
    frame = pd.read_csv(f, sep=';')
    brocade_flowdata = brocade_flowdata.append(frame, ignore_index=True)

# Convert variables to respective type
# cont = continuous
# cat = categorical

#drop na/nan
brocade_flowdata = brocade_flowdata.dropna(axis=0, how='any', subset=['IPV4_SRC_ADDR','IPV4_DST_ADDR'])

cat_cols = ['nfHost','nfSourceID','sequenceNumber','flowSetID','PROTOCOL',
            'TOS', 'TCP_FLAGS','L4_SRC_PORT','IPV4_SRC_ADDR','INPUT_SNMP',
            'L4_DST_PORT','IPV4_DST_ADDR','OUTPUT_SNMP','IPV6_SRC_ADDR',
            'IPV6_DST_ADDR','ICMP_TYPE',
            'DIRECTION','flowId','postNATSourceIPv4Address',
            'postNATDestinationIPv4Address','postNAPTSourceTransportAddress',
            'postNAPTDestinationTransportAddress','firewallEvent',
            'postNATSourceIPv6Address','postNATDestinationIPv6Address',
            'privateEnterpriseNumber','App-ID','User-ID']

cont_cols = ['timeReceived','IN_BYTES',
             'sysUpTime','unixSeconds','FIRST_SWITCHED',
             'LAST_SWITCHED']

for c in cat_cols:
    brocade_flowdata[c] = brocade_flowdata[c].astype('str')

for c in cont_cols:
    brocade_flowdata[c] = brocade_flowdata[c].astype('float64')

# Strip whitespace
brocade_flowdata.rename(columns=lambda x: x.strip(), inplace = True)

print(brocade_flowdata.head(n=10))

def _unique_flow_ids(dataframe):
    # return list of all unique flowId's
    return pd.unique(dataframe.flowId.ravel())

def _flow_groups(dataframe,column):
    groups = dataframe.groupby(column)
    return groups

def _flow_id_dicts(dataframe, column):
    # call unique_flow_ids method
    flow_ids = _unique_flow_ids(dataframe)
    # groupby called for each unique group
    flow_groups = _flow_groups(dataframe, column)
    # create dictionary per flow group
    flow_dicts = {}
    for i in flow_ids:
        flows = []
        flows.append(flow_groups.get_group(i))
        flow_dicts[i] = flows
    return flow_dicts

def _bits_convert(ipv4):
    # convert decimal ip to binary representation
    ipbin = ''.join(['{0:08b}'.format(int(x)) for x in ipv4.split(".")])
    return [float(x) for x in ipbin]

def _prob_matrices(dataframe, column):
    # create empty 32 element array
    all_zeroes = np.zeros((32,), dtype=np.int)
    total = np.outer(all_zeroes, all_zeroes)
    prob = np.outer(all_zeroes, all_zeroes)
    total_flows = len(dataframe)
    flow_dicts = _flow_id_dicts(dataframe, column)

    for k,v in flow_dicts.items():
        for i in v[0].iterrows():
            src_addr = _bits_convert(str(i[1][8]))
            dst_addr = _bits_convert(str(i[1][7]))
            ctm = np.outer(dst_addr,src_addr)
            total = np.add(total,ctm)
        print(total)
        prob = np.divide(total,total_flows)
        #prob = np.round(prob, decimals=3)
        flow_dicts[k] = [v,prob]
    return flow_dicts

# instantiate prob_matrices object
testing = _prob_matrices(brocade_flowdata[0:100], 'flowId')

print(testing)


