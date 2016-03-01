import collections


class updateOpsMeta(type):
    registry = {}

    def __new__(cls, name, bases, attrs):
        new_cls = type.__new__(cls, name, bases, attrs)
        cls.registry[new_cls.__name__] = new_cls().op
        return new_cls

class updateOps(object):
    __metaclass__ = updateOpsMeta

    def op(self, node, uPkg):
        pass


class updatePkg(object):

    def __init__(self, name = "value", op = "vAdd", value = 1, sample = None):

        self.name = name
        self.op = updateOps.registry[op]
        self.value = value
        self.sample = sample

    def __getattr__(self, field):
        return self.sample[field]

class vAdd(updateOps):

    def op(self, node, upkg):

        while node:
            node.value += upkg.value
            node = node.parent

class hAdd(updateOps):

    def op(self, node, upkg):

        vup = updatePkg(sample = upkg.sample)

        while node:
            node.stats[upkg.name].update(vup)
            node = node.parent


class hAddFlow(updateOps):

    flows ={}
    def op(self, node, upkg):

            if upkg.name not in self.flows:
                self.flows[upkg.name] = collections.defaultdict(set)

            if upkg.flowId not in self.flows[upkg.name][node.name]:
                self.flows[upkg.name][node.name].add(upkg.flowId)

                vup = updatePkg(sample = upkg.sample)
                while node:
                    node.stats[upkg.name].update(vup)
                    node = node.parent













