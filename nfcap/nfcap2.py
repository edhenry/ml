import socket, select, struct, collections, ipaddress, nfTypes, ctypes, time, os, csv, yaml, hiyapyco, profile


fileWindow = 3600
fileAgeOut = 86400

fieldIndex = {'postNAPTSourceTransportAddress': 27, 'TOS': 10, 'postNATSourceIPv6Address': 30, 'IPV4_DST_ADDR': 16, 'INPUT_SNMP': 14, 'User-ID': 34, 'LAST_SWITCHED': 18, 'OUTPUT_SNMP': 17, 'DIRECTION': 23, 'IN_PKTS': 8, 'postNATDestinationIPv6Address': 31, 'FIRST_SWITCHED': 19, 'postNATSourceIPv4Address': 25, 'IPV6_DST_ADDR': 21, 'postNATDestinationIPv4Address': 26, 'TCP_FLAGS': 11, 'postNAPTDestinationTransportAddress': 28, 'PROTOCOL': 9, 'L4_SRC_PORT': 12, 'ICMP_TYPE': 22, 'IN_BYTES': 7, 'App-ID': 33, 'IPV4_SRC_ADDR': 13, 'flowId': 24, 'IPV6_SRC_ADDR': 20, 'firewallEvent': 29, 'privateEnterpriseNumber': 32, 'L4_DST_PORT': 15}
dfnames = [
    "timeReceived",
    "nfHost",
    "nfSourceID",
    "sysUpTime",
    "unixSeconds",
    "sequenceNumber",
    "flowSetID",
    "IN_BYTES",
    "IN_PKTS",
    "PROTOCOL",
    "TOS",
    "TCP_FLAGS",
    "L4_SRC_PORT",
    "IPV4_SRC_ADDR",
    "INPUT_SNMP",
    "L4_DST_PORT",
    "IPV4_DST_ADDR",
    "OUTPUT_SNMP",
    "LAST_SWITCHED",
    "FIRST_SWITCHED",
    "IPV6_SRC_ADDR",
    "IPV6_DST_ADDR",
    "ICMP_TYPE",
    "DIRECTION",
    "flowId",
    "postNATSourceIPv4Address",
    "postNATDestinationIPv4Address",
    "postNAPTSourceTransportAddress",
    "postNAPTDestinationTransportAddress",
    "firewallEvent",
    "postNATSourceIPv6Address",
    "postNATDestinationIPv6Address",
    "privateEnterpriseNumber",
    "App-ID",
    "User-ID"
]

    

dfnames = [
    "timeReceived",
    "nfHost",
    "nfSourceID",
    "sysUpTime",
    "unixSeconds",
    "sequenceNumber",
    "flowSetID",
    "IN_BYTES",
    "IN_PKTS",
    "PROTOCOL",
    "TOS",
    "TCP_FLAGS",
    "L4_SRC_PORT",
    "IPV4_SRC_ADDR",
    "INPUT_SNMP",
    "L4_DST_PORT",
    "IPV4_DST_ADDR",
    "OUTPUT_SNMP",
    "LAST_SWITCHED",
    "FIRST_SWITCHED",
    "IPV6_SRC_ADDR",
    "IPV6_DST_ADDR",
    "ICMP_TYPE",
    "DIRECTION",
    "flowId",
    "postNATSourceIPv4Address",
    "postNATDestinationIPv4Address",
    "postNAPTSourceTransportAddress",
    "postNAPTDestinationTransportAddress",
    "firewallEvent",
    "postNATSourceIPv6Address",
    "postNATDestinationIPv6Address",
    "privateEnterpriseNumber",
    "App-ID",
    "User-ID"
]


csvDirectory = "/home/dwinkwor/nfcap/logs/csv"
    
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind(('10.254.11.245', 2055))
readList = [server]

def fuzzCSVPath(nfHost, nfSourceID, subdir):

    sDir = csvDirectory + "/" + nfHost.replace(".","-") + "/" + str(nfSourceID) + "/" + subdir
    
    if not os.access(sDir, os.F_OK):
        os.makedirs(sDir)

    if os.access(sDir, os.F_OK|os.W_OK):
        return sDir

class keyedStruct(ctypes.BigEndianStructure):
    _pack_ = 1
    
    def __getitem__(self, name):
        return object.__getattribute__(self, name)
        
        
class nfHeader(ctypes.BigEndianStructure):
    _fields_ = [
        ("version", ctypes.c_uint16),
        ("count", ctypes.c_uint16),
    ]

class nf9Header(ctypes.BigEndianStructure):
    _fields_ = [
        ("sysUpTime", ctypes.c_uint32),
        ("unixSeconds", ctypes.c_uint32),
        ("sequenceNumber", ctypes.c_uint32),
        ("sourceID", ctypes.c_uint32)
    ]
    
class flowSet(ctypes.BigEndianStructure):
    _fields_ = [
        ("ID", ctypes.c_uint16),
        ("length", ctypes.c_uint16)]

class templateHeader(ctypes.BigEndianStructure):
    _fields_ = [
        ("ID", ctypes.c_uint16),
        ("fieldCount", ctypes.c_uint16)
    ]

class fieldSpec(ctypes.BigEndianStructure):
    _fields_ = [
        ("fieldType", ctypes.c_uint16),
        ("fieldLength", ctypes.c_uint16)
    ]


class nf9template(object):

    ## Instances of this class are created for incoming template records
    ##
    ## self.templateID
    ## ---------------
    ##     Description:  The template ID is a number >= 256 per RFC 3954.  
    ##     Purpose:  Each observation domain has an associated set of 
    ##               templates used to export different kinds of flows.
    ##               The template ID uniquely identifies a template 
    ##               within an observation domain.
    ##
    ## self.fieldCount
    ## ---------------
    ##     Description:  The number of fields in the template
    ##
    ## self.fieldTypeObjects
    ## -----------------
    ##     Description:  An ordered list of objects associated with the field
    ##                   types specified in the template record.
    ##     Purpose:  The classes for these objects are specified in nfTypes.py.  
    ##               They represent the Field Types specified in section 8 of
    ##               RFC 3954.  These objects do not hold the actual field
    ##               values.  They are just containers of meta-data and helper
    ##               functions for the field types and values.
    ##               
    ## self.length
    ## -----------
    ##     Description:  The length, in bytes, of a dataflow using this template.
    ##     Purpose:  Before attempting to parse incoming dataflows using this 
    ##               template, their length will be validated against this value.
    ##                
    ## self.struct
    ## -----------
    ##     Description:  A ctypes Big Endian structure class with members corresponding
    ##                   to the fields in the template record.
    ##     Purpose:  This struct is used to parse raw packet buffer data and return a structure
    ##               instance with flow data.  
    ##
    ## self.sig
    ## --------
    ##     Description:  A hash generated over the entirety of the template record.
    ##     Purpose:  This hash is used to compare instances of nf9templates.  If
    ##               both instances have the same template ID, field count, fields, 
    ##               and field lengths, they are considered equal.
    
    def __init__(self,data,offset):
        
        ## Parse the template record header to get the template ID and field count
        th = templateHeader.from_buffer(data, offset)
        
        
        self.templateID = th.ID
        self.fieldCount = th.fieldCount
        self.fieldTypeObjects = []
        self.length = 0
        
        fieldList = []
        hlist = [(self.templateID,self.fieldCount)]
        
        ##Iterate over the fields specified in the template record
        for x in xrange(1, self.fieldCount + 1):
            offset += 4
            
            ## Parse the field spec to get the field type and field length
            z = fieldSpec.from_buffer(data, offset)
            
            ## Increment self.length by the field length
            self.length += z.fieldLength
            
            ## Create a python class of the appropriate field type and length
            m = nfTypes.metaIE.registry[z.fieldType](z.fieldLength)
            
            self.fieldTypeObjects.append(m)
            fieldList.append((m.nf9name, m.seeType))
            hlist.append((z.fieldType,z.fieldLength))
        
        ## Create a ctypes structure for parsing of data flows using this template        
        self.struct = type(str(self.templateID), (keyedStruct,), {"_fields_": fieldList})
        
        ## Create a hash signature of the template record for comparing with other nf9template instances
        self.sig = hash(tuple(hlist))
    

    ## For the comments assume there is an nf9template instance called "temp"
    
    def __call__(self, data, offset):
        ## Makes an instance of nf9template callable.
        ## 'temp(data, offset)' parses an incoming dataflow and returns a ctypes structure
        return self.struct.from_buffer(data, offset)
    
    def __iter__(self):
        ## Makes an nf9template iterable.
        ## 'for ftype in temp:'  will iterate over the field type objects of this template
        for item in self.fieldTypeObjects:
            yield item
    
    def __getitem__(self, index):
        ## 'temp[n]' will return the field type object of field n.
        return self.fieldTypeObjects[index]
    
    def __len__(self):
        return self.length
    
    def __repr__(self):
        ## The template ID is the official string representation of an nf9template instance.
        ## If using an nf9template instance as a key in a dictionary(i.e., 'myDict[temp]'),
        ## the key will appear as the template ID.
        return str(self.templateID)
    
    def __eq__(self, other):
        ## When comparing an nf9instance to another nf9instance using "==" or "eq", self.sig
        ## is used.  Otherwise, self.templateID is used.
        if isinstance(other, nf9template):
            return self.sig == other.sig
        else:
            return self.templateID == other
            
    def __hash__(self):
        ## Returns a hash of the template ID.
        return hash(self.templateID)

        
        
    
class nf9observationDomain(object):

    ## This class is instantiated for each observation domain that each
    ## flow exporter is exporting flows for.  
    ## 
    ## From RFC 3954:
    ##     NetFlow Collectors SHOULD use the combination of the source IP
    ##     address and the Source ID field to separate different export
    ##     streams originating from the same Exporter.


    ## templateCache holds the template definitions exported by the
    ## devices exporting netflows.  It is indexed by the template ID.
    
    templateCache = {}

    def __init__(self,domain):
        self.domainID = domain
        nf9observationDomain.domainRegistry[domain] = self
        self.templates = {}
        self.templateFile = {"timeStamp": 0, "name": ""}
        self.dFileStamp = 0
        self.dFileName = ""
        self.lastTimeStamp = None
        
    def logFlows(self, templateList = [], flowList = []):
    
        modeParam = os.F_OK | os.R_OK
        

        if templateList:
            nfHost = templateList[0]["nfHost"]
            nfSourceID = templateList[0]["nfSourceID"]
            timeReceived = templateList[0]["timeReceived"]
            tPath = fuzzCSVPath(nfHost, nfSourceID, "templates")
            newFile = False
            if tPath:
                if timeReceived - self.templateFile["timeStamp"] > fileWindow:               
                    self.templateFile["timeStamp"] = timeReceived
                    self.templateFile["name"] = str(timeReceived) + ".csv"
                    newFile = True
                
                with open(tPath + "/" + self.templateFile["name"], 'a+') as csvfile:

                    twriter = csv.DictWriter(csvfile, delimiter=";", fieldnames=templateList[0].keys())
                    if newFile:
                        twriter.writeheader()
                        newFile = False
                    twriter.writerows(templateList)
                        
        if flowList:
            nfHost = flowList[0][1]
            nfSourceID = flowList[0][2]
            timeReceived = flowList[0][0]
            tPath = fuzzCSVPath(nfHost, nfSourceID, "dataFlows")
            newFile = False
            if tPath:
                if timeReceived - self.dFileStamp > fileWindow:               
                    self.dFileStamp = timeReceived
                    self.dFileName = str(int(timeReceived)) + ".csv"
                    newFile = True
                
                with open(tPath + "/" + self.dFileName, 'a+') as csvfile:

                    twriter = csv.writer(csvfile, delimiter=";")
                    if newFile:
                        twriter.writerow(dfnames)
                        newFile = False
                    twriter.writerows(flowList)
                            
                    
    def parseTemplate(self, data, offset):
        
        ## Instantiate an nf9template instance from the incoming template record.
        ## Then check if it's a new or updated template by looking for 
        ## the template ID in this observation domain's template 
        ## cache.  If it exists in the cache, then compare the new template's
        ## sig with the existing template's sig.  If they
        ## are the same, then do nothing and return nothing.
        ##
        ## If they are different, or if the template ID isn't in the cache
        ## then update the templateCache.
        
        newt = nf9template(data, offset)
        
        if newt in self.templateCache:
            if newt == self.templateCache[newt]:
                return
        
        self.templateCache[newt] = newt
        
        return newt.templateID
        
    def parseData(self, data, offset, fSetID, flowEntry):
        
        dFlow = self.templateCache[fSetID](data, offset)

        for fieldType in self.templateCache[fSetID]:
            flowEntry[fieldIndex[fieldType.nf9name]] = fieldType.getPyVal(dFlow[fieldType.nf9name])
        
    def parseNew(self, data, header, count):
        offset = 20
        flowList = []
        templateList = []
        timeReceived = time.time()
        for x in xrange(1, count + 1):
            y = flowSet.from_buffer(data, offset)
            if y.ID == 0:
                updatedID = self.parseTemplate(data, offset + 4)
                if updatedID:
                    templateEntry = collections.OrderedDict()
                    templateEntry["timeReceived"] = timeReceived
                    templateEntry["nfHost"] = self.domainID[0]
                    templateEntry["nfSourceID"] = self.domainID[1]
                    templateEntry["sysUpTime"] = header.sysUpTime
                    templateEntry["unixSeconds"] = header.unixSeconds
                    templateEntry["sequenceNumber"] = header.sequenceNumber
                    templateEntry["templateID"] = updatedID
                    templateEntry["fields"] = self.templates[updatedID]["yaml"]
                    templateList.append(templateEntry)
            elif y.ID > 255 and y.ID in self.templateCache:
                if y.length == len(self.templateCache[y.ID]):
                    flowEntry = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
                    flowEntry[0] = timeReceived
                    flowEntry[1] = self.domainID[0]
                    flowEntry[2] = self.domainID[1]
                    flowEntry[3] = header.sysUpTime
                    flowEntry[4] = header.unixSeconds
                    flowEntry[5] = header.sequenceNumber
                    flowEntry[6] = y.ID
                    self.parseData(data, offset + 4, y.ID, flowEntry)
                    flowList.append(flowEntry)
            offset += y.length
        
        self.logFlows(templateList, flowList)
      
running = 1

while running:
    inputReady, outputReady, exceptReady = select.select(readList, [], [])
    
    for item in inputReady:
        
        data = bytearray(8192)
        nbytes, addr = item.recvfrom_into(data)
        if data:
            xhead = nfHeader.from_buffer(data)
            if xhead.version == 9:
                yhead = nf9Header.from_buffer(data,4)
                domain = (addr[0], yhead.sourceID)
                if domain not in nf9observationDomain.domainRegistry:
                    nf9observationDomain.domainRegistry[domain] = nf9observationDomain(domain)
                
                nf9observationDomain.domainRegistry[domain].parseNew(data, yhead, xhead.count)
                

                
                 
    
            
    

    