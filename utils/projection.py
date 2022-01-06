import json
import os
import collections


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


class OntologyProjection:
    def __init__(self, root) -> None:
        ontology = json.load(open(os.path.join(root, 'ontology.json'), 'r', encoding='utf-8'))
        entities = {}
        invert_docs = {}
        for slot, values in ontology['slots'].items():
            if isinstance(values, str):
                with open(os.path.join(root, values), "r", encoding="utf-8") as fi:
                    values = [x.strip() for x in fi]
            entities[slot] = set(values)
            invert_doc = collections.defaultdict(set)
            for entity in entities[slot]:
                for ch in entity:
                    invert_doc[ch].add(entity)
            invert_docs[slot] = invert_doc
        self.entities = entities
        self.invert_docs = invert_docs

    def projection(self, slot, val):
        if val in self.entities[slot] or not val:
            return val
        mini = None
        mind = 16777215, 256
        inv_dict = self.invert_docs[slot]
        for v in set.union(*(inv_dict[ch] for ch in val)):
            kou = levenshteinDistance(v, val), abs(len(v) - len(val))
            if kou < mind:
                mind = kou
                mini = v
        if mind >= (len(val) * 0.5, 256):
            return None
        return mini
