import json
import numpy as np
from collections import defaultdict

class RuleWriter(object):
    """
    This class is for writing rules in a format 
    the judging software can read
    Usage might look like this:

    rule_writer = RuleWriter()
    for lhs, rhs, prob in out_rules:
        rule_writer.add_rule(lhs, rhs, prob)
    rule_writer.write_rules()
    """

    def __init__(self):
        self.rules = []

    def add_rule(self, lhs, rhs, prob):
        """Add a rule to the list of rules
        Does some checking to make sure you are using the correct format.

        Args:
            lhs (str): The left hand side of the rule as a string
            rhs (Iterable(str)): The right hand side of the rule. 
                Accepts an iterable (such as a list or tuple) of strings.
            prob (float): The conditional probability of the rule.
        """
        assert isinstance(lhs, str)
        assert isinstance(rhs, list) or isinstance(rhs, tuple)
        assert not isinstance(rhs, str)
        nrhs = []
        for cl in rhs:
            assert isinstance(cl, str)
            nrhs.append(cl)
        assert isinstance(prob, float)

        self.rules.append((lhs, nrhs, prob))

        
    def write_rules(self, filename="q1.json"):
        """Write the rules to an output file.

        Args:
            filename (str, optional): Where to output the rules. Defaults to "q1.json".
        """
        json.dump(self.rules, open(filename, "w"))


def form_relation(parent, children):
    """
    This is a recurrent functon to form relations
    Given : Parent and list of immediate children as arguments
    Relations are stores in rel_dict global Dictionary
    :param parent: String
    :param children: List of strings
    """

    # Check if the children is a Terminal, store/ create a relation between Parent -> Children i.e Non Terminal -> Terminal
    if (isinstance(children, list) == False):
        if parent in rel_dict:
            rel_dict[parent].append([children])
        else:
            rel_dict[parent] = [[children]]

    else:
        # If not, for each terminal/ non terminal in the list of children
        # * append them to RHS
        # * Run a recurrent function call based on the number of their children
        rhs = []
        for child in children:
            rhs.append(child[0])
            if(len(child)==2 and isinstance(child[1], list) == False):
                form_relation(child[0], str(child[1]))
            else:
                form_relation(child[0], child[1:])

        if parent in rel_dict:
            #print(rhs)
            rel_dict[parent].append(rhs)
        else:
            #print(rhs)
            rel_dict[parent] = [rhs]

def calculate_prob(rel_dict, rule_writer):
    """
    This function calculates the probabilities of the relation
    Given Relations as dictionary i.e of format LHS: {[RHS], [RHS]} where RHS is a List
    rule_writer to add rules
    :param rel_dict: Dictionary of format key: value  where value is {[v_i],...,[v_j]} and v_x is a list
    :param rule_writer:
    """

    # For each LHS, find
    # Number of unique RHS and their frequency
    for lhs in rel_dict.keys():
        rhs = rel_dict[lhs]
        # Initialize dict
        prob_map = {}

        # Use Append through Iteration to find the frequency
        for e in rhs:
            prob_map.setdefault(tuple(e), list()).append(1)

        # With frequency, find the Conditional Probability
        # len(rhs) gives us the number of occurances of LHS
        for key, val in prob_map.items():
            prob_map[key] = sum(val)/len(rhs)

        # Write the rules
        for rhs, prob in prob_map.items():
            rule_writer.add_rule(lhs, rhs, prob)
            #print("Alpha: " + key + " Beta: " + e[0] + " Prob: " + str(e[1]))


def main():
    # load the parsed sentences
    psents = json.load(open("parsed_sents_list.json", "r"))
    #psents = [['A', ['B', ['C', 'blue']], ['B', 'cat']]] # test case

    # print a few parsed sentences
    # NOTE: you can remove this if you like
    for sent in psents:

        # Assigning Parent to LHS, and passing the parents and list of children as arguments
        lhs = sent[0]
        form_relation(lhs, sent[1:])

    #print(rel_dict)

    # Write the rules to the correct output file using the write_rules method, after calculating the probabilities
    rule_writer = RuleWriter()
    calculate_prob(rel_dict, rule_writer)
    rule_writer.write_rules()

if __name__ == "__main__":
    rel_dict = {}
    main()