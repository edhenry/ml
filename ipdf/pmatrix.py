import nodestats, ipdf, csv

class pshell(ipdf.pmfShell):

    def do_load(self, arg):
        od = ipdf.statsPMF()

        with open(arg,'rb') as csvFile:
            csvReader = csv.DictReader(csvFile, delimiter=';')
            print "Loading file...\n"
            for flow in csvReader:
                src = flow["IPV4_SRC_ADDR"]
                dstPort = int(flow["L4_DST_PORT"])
                nat = flow["postNATSourceIPv4Address"]
                if not nat and src and dstPort == 53:
                      upkg = nodestats.updatePkg(name="toDNS", op="hAddFlow", sample=flow)
                      od.update(upkg)

        self.pmf = od
        self.cwn = self.pmf.root

if __name__ == "__main__":
      pshell().cmdloop()
