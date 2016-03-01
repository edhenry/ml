import csv, sys, ipaddress, cmd, os, readline, re, nodefinder, nodestats, collections



class hNode(object):

    def __init__(self, nodeID = None, parent=None):

        self.parent = parent
        self.nodeID = nodeID
        self.leaf = False
        self.children = {}
        self.toponame = False
        self.root = False

        if self.parent:
            if self.parent.root:
                self.name = "/" + str(self.nodeID)
            else:
                if self.parent.parent.root:
                    self.name = self.parent.name + "/" + str(self.nodeID)
                else:
                    self.name = self.parent.name + "." + str(self.nodeID)
        else:
            self.name = "/"
            self.root = True

    def updateTree(self, nlist):

        if len(nlist) > 0:
            newchild = False
            n = nlist[0]
            if n not in self.children:
                self.children[n] = hNode(n, self)
                newchild = True

            newchildren = self.children[n].updateTree(nlist[1:])
            if newchild:
                    newchildren.append(self.children[n])
            return newchildren
        else:
            self.leaf = True
            return [self]


class PMF(object):

    def __init__(self, unit = "octets"):

        self.convertIP = self.octetsConvert
        if unit == "bits":
            self.convertIP = self.bitsConvert
        self.root = hNode()
        self.cwn = self.root
        self.getNode = nodefinder.nodeFinder()
        self.allNodes = {}
        self.segmentLookup = {}


    def cn(self, path):

        if path:
            self.cwn = self.getNode(path, self.root, self.cwn)

        return self.cwn

    def ln(self, path):

        n = self.cwn

        if path:
            n = self.getNode(path, self.root, self.cwn)

        return n.name, n.children

    def getCurrentNode(self):
        return self.cwn

    def octetsConvert(self, ipv4):
        return [int(x) for x in ipv4.split('.') if x]

    def bitsConvert(self,ipv4):
        y = bin(int(ipaddress.IPv4Address(unicode(ipv4))))
        return [int(x) for x in y[2:]]

class valuePMF(PMF):

    def __init__(self, units="octets"):
        PMF.__init__(self, units)
        self.root.value = 0

    def update(self, upkg):

        egress = upkg.OUTPUT_SNMP
        dst = upkg.IPV4_DST_ADDR

        tpath = "/" + egress
        path = tpath + "/" + dst


        if path not in self.allNodes:
            nlist = [int(egress)] + self.convertIP(dst)
            nodes = self.root.updateTree(nlist)
            for node in nodes:
                self.allNodes[node.name] = node
                node.value = 0
            self.segmentLookup[dst] = self.allNodes[tpath]
        n = self.allNodes[path]

        upkg.op(n, upkg)

class statsPMF(PMF):

    def __init__(self, units="octets"):
        PMF.__init__(self, units)
        self.root.stats = collections.defaultdict(valuePMF)

    def update(self, upkg):

        ingress = upkg.INPUT_SNMP
        src = upkg.IPV4_SRC_ADDR

        tpath = "/" + ingress
        path = tpath + "/" + src

        if path not in self.allNodes:
            nlist = [int(ingress)] + self.convertIP(src)
            nodes = self.root.updateTree(nlist)
            for node in nodes:
                self.allNodes[node.name] = node
                node.stats = collections.defaultdict(valuePMF)
            self.segmentLookup[src] = self.allNodes[tpath]

        n = self.allNodes[path]

        upkg.op(n, upkg)

    def ns(self, path):

        n = self.cwn

        if path:
            n = self.getNode(path, self.root, self.cwn)

        topstats = {}
        for tree in n.stats:
            topstats[tree] = n.stats[tree].root.value

        return (n.name, topstats)

    def lnstat(self, stat):

        node = self.cwn

        return node.stats[stat].allNodes.keys()


####  Everything below is the interactive shell used to interact
###   with the data structure.


class pmfShell(cmd.Cmd):

    def __init__(self):
        cmd.Cmd.__init__(self)
        self.intro = "Welcome to the IPv4 Octet PMF shell.  Type help or ? to list commands.\n"
        self.prompt = '> '
        self.pmf = statsPMF()

    def do_load(self, arg):
        pass;

    def do_ln(self, arg):
        "Lists children of the current node"

        response = self.pmf.ln(arg)

        print "Node: " + response[0] + "\n"

        for child in response[1]:
            print child

    def do_cn(self, arg):
        "Change working node."

        n = self.pmf.cn(arg)
        self.prompt = n.name + "> "
        print "Current working node: " + n.name

    def do_ns(self, arg):
        "Prints stats of current node, or the provided child node."

        response = self.pmf.ns(arg)

        print "Node: " + response[0] + "\n"

        stats = response[1]
        for name in stats:
            print name + ": " + str(stats[name])

    def do_lnstat(self, arg):

        response = self.pmf.lnstat(arg)

        for node in response:
            print node

    def do_quit(self, arg):
        "Quit."
        print "Exiting..."
        sys.exit(1)









