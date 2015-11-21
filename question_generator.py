"""
A question generator.
Author: Mengye Ren  Email: mren@cs.toronto.edu

Usage:
    python question_generator.py -parser_path {Stanford parser path} \
                                 -sentence {Single sentence} \
                                 -list {List of sentences} \
                                 -parsed_file {Parsed file} \
                                 -output {Output file}

Prerequisites:
    1. You need to download NLTK WordNet package.
    >> python
    >> import nltk
    >> nltk.download()
    >> d
    >> wordnet

    2. You need to download Stanford Parser
    at http://nlp.stanford.edu/software/lex-parser.shtml#Download
    Extract the zip into a folder and remember the path

    3. You need to copy lexparser_sentence.sh into the Stanford Parser folder.
    >> cp lexparser_sentence.sh stanford-parser/lexparser_sentence.sh

Examples:
    1. Run single sentence.
    >> python question_generator.py -sentence "A man is riding a horse"

    2. Run a list of sentences.
    Provide a file with each line in the file to be a sentence.
    Output is a pickle file, storing a list. Each element in the list is a
    tuple of five fields:
        (1) Original sentence ID (0-based)
        (2) Original sentence
        (3) Generated question
        (4) Answer to the generated question
        (5) Type of the generated question

    >> python question_generator.py -list sentences.txt -output questions.pkl

    3. Run a pre-parsed file.
    Run stanford parser to pre-compute the parse trees.

    >> lexparser.sh sentences.txt > sentences_parsed.txt
    >> python question_generator.py -list sentences.txt \
                                    -parsed_file sentences_parsed.txt \
                                    -output questions.pkl
"""

from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import argparse
import copy
import cPickle as pkl
import logger
import os
import re
import subprocess
import sys
import time

# A white list for color adjectives.
whiteListColorAdj = set(['red', 'yellow', 'orange', 'brown', 'green', 'blue',
                         'purple', 'black', 'white', 'gray', 'grey', 'violet'])

# A white list for lex names that can appear in object type questions.
whiteListLexname = set(['noun.animal', 'noun.artifact', 'noun.food',
                        'noun.object', 'noun.plant', 'noun.possession',
                        'noun.shape'])

# A black list for nouns with color.
blackListColorNoun = set(['ride', 'riding', 'past', 'stand', 'standing',
                          'eating', 'holding', 'frosting', 'glow', 'glowing',
                          'resting', 'parked'])

# A black list for nouns to appear in object type questions.
blackListNoun = set(['female', 'females', 'male', 'males', 'commuter',
                     'commuters', 'player', 'players', 'match', 'matches',
                     'rider', 'riders', 'doll', 'dolls', 'ride', 'rides',
                     'riding', 'past', 'pasts', 'teddy', 'fan', 'fans',
                     'street', 'streets', 'arm', 'arms', 'head', 'heads',
                     'slope', 'slopes', 'shoot', 'shoots', 'photo', 'photos',
                     'space', 'spaces', 'stand', 'stands', 'standing',
                     'cross', 'crosses', 'crossing', 'eating', 'walking',
                     'driving', 'upright', 'structure', 'turn', 'system',
                     'arrangement', 'set', 'top', 'while', 'well', 'area',
                     'produce', 'thing', 'things', 'cut', 'cuts', 'holding',
                     'frosting', 'glow', 'glowing', 'ground', 'parked'])

# A black list for compound nouns that appear in object type questions.
blackListCompoundNoun = set(['tennis', 'delivery', 'soccer', 'baseball',
                             'fighter', 'mother', 'window'])

# A black list of verbs that should never be parsed as verbs.
blackListVerb = set(['sink', 'sinks', 'counter', 'counters', 'cupboard',
                     'cupboards', 'has', 'have', 'contain', 'contains',
                     'containing', 'contained', 'spaniel', 'spaniels',
                     'mirror', 'mirrors', 'shower', 'showers', 'stove',
                     'stoves', 'bowl', 'bowls', 'tile', 'tiles', 'mouthwash',
                     'mouthwashes', 'smoke', 'smokes'])

# A black list of prepositions that we avoid asking questions within the
# clause.
blackListPrep = set(['with', 'of', 'in', 'down', 'as'])

# A black list of locations that we avoid asking location type questions upon.
blackListLocation = set(['t-shirt', 't-shirts', 'jeans', 'shirt', 'shirts',
                         'uniform', 'uniforms', 'jacket', 'jackets', 'dress',
                         'dresses', 'hat', 'hats', 'tie', 'ties', 'costume',
                         'costumes', 'attire', 'attires', 'match', 'matches',
                         'coat', 'coats', 'cap', 'caps', 'gear', 'gears',
                         'sweatshirt', 'sweatshirts', 'helmet', 'helmets',
                         'clothing', 'clothings', 'cloth', 'clothes',
                         'blanket', 'blankets', 'enclosure', 'enclosures',
                         'suit', 'suits', 'photo', 'photos', 'picture',
                         'pictures', 'round', 'rounds', 'area', 'well',
                         'skirt', 'snowsuit', 'sunglasses', 'sweater', 'mask',
                         'frisbee', 'frisbees', 'shoe', 'umbrella', 'towel',
                         'scarf', 'phone', 'cellphone', 'motorcycle',
                         'device', 'computer', 'cake', 'hydrant', 'desk',
                         'stove', 'sculpture', 'lamp', 'fireplace', 'bags',
                         'laptop', 'trolley', 'toy', 'bus', 'counter',
                         'buffet', 'engine', 'graffiti', 'clock', 'jet',
                         'ramp', 'brick', 'taxi', 'knife', 'flag', 'screen',
                         'parked'])

# A black list of verbs that are not asked in the location type questions.
blackListVerbLocation = set(['sink', 'sinks', 'counter', 'counters',
                             'cupboard', 'cupboards', 'has', 'have',
                             'contain', 'contains', 'containing', 'contained',
                             'can', 'cans'])

# A black list of nouns that are not asked in the how many type questions.
blackListNumberNoun = set(['pole', 'vase', 'kite', 'hay', 'shower', 'paddle',
                           'buffet', 'bicycle', 'bike', 'elephants'])

# A dictionary of synonyms to convert to.
synonymConvert = {'busses': 'buses', 'plane': 'airplane',
                  'planes': 'airplanes', 'aircraft': 'airplane',
                  'aircrafts': 'airplane', 'jetliner': 'airliner',
                  'jetliners': 'airliners', 'bike': 'bicycle',
                  'bikes': 'bicycles', 'cycle': 'bicycle',
                  'cycles': 'bicycles', 'motorbike': 'motorcycle',
                  'motorbikes': 'motorcycles', 'grey': 'gray',
                  'railroad': 'rail', 'cell': 'cellphone',
                  'doughnut': 'donut', 'doughnuts': 'donuts'}

# Compound nouns
compoundNoun = set(['fighter jet', 'soccer ball', 'tennis ball'])

# Special characters that may appear in the text.
charText = set(['.', ',', '-', '\'', '`', '/', '>', ':', ';', '\\', '!', '?',
                '&', '-', '=', '#', '$', '@', '_', '*', '+', '%', chr(194),
                chr(160)])

# Special characters that may appear in the class name.
charClassName = set(['.', ',', '$', '\'', '`', ':', '-', '#'])

# WordNet lemmatizer.
lemmatizer = WordNetLemmatizer()

# Logger
log = logger.get()


class TreeNode:
    """Parse tree.
    """

    def __init__(self, className, text, children, level):
        """Construct a tree.
        """
        self.className = className
        self.text = text
        self.children = children
        self.level = level
        pass

    def __str__(self):
        """To string (with tree structure parentheses).
        """
        strlist = []
        for i in range(self.level):
            strlist.append('    ')
        strlist.extend(['(', self.className])
        if len(self.children) > 0:
            strlist.append('\n')
            for child in self.children:
                strlist.append(child.__str__())
            if len(self.text) > 0:
                for i in range(self.level + 1):
                    strlist.append('    ')
            else:
                for i in range(self.level):
                    strlist.append('    ')
        else:
            strlist.append(' ')
        strlist.append(self.text)
        strlist.append(')\n')
        return ''.join(strlist)

    def toSentence(self):
        """Unfold the tree structure into a string.
        """
        strlist = []
        for child in self.children:
            childSent = child.toSentence()
            if len(childSent) > 0:
                strlist.append(childSent)
        if len(self.text) > 0:
            strlist.append(self.text)
        return ' '.join(strlist)

    def relevel(self, level):
        """Re-assign level.
        """
        self.level = level
        for child in self.children:
            child.relevel(level + 1)

    def copy(self):
        """Clone a tree.
        """
        children = []
        for child in self.children:
            children.append(child.copy())
        return TreeNode(self.className, self.text, children, self.level)


class TreeParser:
    """Finite state machine implementation of syntax tree parser.
    """

    def __init__(self):
        self.state = 0
        self.currentClassStart = 0
        self.currentTextStart = 0
        self.classNameStack = []
        self.childrenStack = [[]]
        self.root = None
        self.rootsList = []
        self.level = 0
        self.stateTable = [self.state0, self.state1, self.state2,
                           self.state3, self.state4, self.state5, self.state6]
        self.raw = None
        self.state = 0

    def parse(self, raw):
        if not self.isAlpha(raw[0]):
            self.raw = raw
            for i in range(len(raw)):
                self.state = self.stateTable[self.state](i)

    @staticmethod
    def isAlpha(c):
        return 65 <= ord(c) <= 90 or 97 <= ord(c) <= 122

    @staticmethod
    def isNumber(c):
        return 48 <= ord(c) <= 57

    @staticmethod
    def exception(raw, i):
        print raw
        raise Exception(
            'Unexpected character "%c" (%d) at position %d'
            % (raw[i], ord(raw[i]), i))

    @staticmethod
    def isClassName(s):
        if TreeParser.isAlpha(s) or s in charClassName:
            return True
        else:
            return False

    @staticmethod
    def isText(s):
        if TreeParser.isAlpha(s) or TreeParser.isNumber(s) or s in charText:
            return True
        else:
            return False

    def state0(self, i):
        if self.raw[i] == '(':
            return 1
        else:
            return 0

    def state1(self, i):
        if self.isClassName(self.raw[i]):
            self.currentClassStart = i
            self.level += 1
            self.childrenStack.append([])
            return 2
        else:
            self.exception(self.raw, i)

    def state2(self, i):
        if self.isClassName(self.raw[i]):
            return 2
        else:
            self.classNameStack.append(self.raw[self.currentClassStart:i])
            if self.raw[i] == ' ' and self.raw[i + 1] == '(':
                return 0
            elif self.raw[i] == ' ' and self.isText(self.raw[i + 1]):
                return 4
            elif self.raw[i] == '\n':
                return 3
            else:
                self.exception(self.raw, i)

    def state3(self, i):
        if self.raw[i] == ' ' and self.raw[i + 1] == '(':
            return 0
        elif self.raw[i] == ' ' and self.raw[i + 1] == ' ':
            return 3
        elif self.raw[i] == ' ' and self.isText(self.raw[i + 1]):
            return 4
        else:
            return 3

    def state4(self, i):
        if self.isText(self.raw[i]):
            # global currentTextStart
            self.currentTextStart = i
            return 5
        else:
            self.exception(self.raw, i)

    def state5(self, i):
        if self.isText(self.raw[i]):
            return 5
        elif i == len(self.raw) - 1:
            return 5
        elif self.raw[i] == ')':
            self.wrapup(self.raw[self.currentTextStart:i])
            if self.level == 0:
                return 0
            elif self.raw[i + 1] == ')':
                return 6
            else:
                return 3
        else:
            self.exception(self.raw, i)

    def state6(self, i):
        if self.level == 0:
            return 0
        elif self.raw[i] == ')':
            self.wrapup('')
            return 6
        else:
            return 3

    def wrapup(self, text):
        self.level -= 1
        root = TreeNode(self.classNameStack[-1], text,
                        self.childrenStack[-1][:], self.level)
        del self.childrenStack[-1]
        del self.classNameStack[-1]
        self.childrenStack[-1].append(root)
        if self.level == 0:
            self.rootsList.append(root)


class QuestionGenerator:
    """Generates 4 types of questions.
    """

    def __init__(self):
        self.lexnameDict = {}
        pass

    @staticmethod
    def escapeNumber(line):
        """Convert numbers into English.
        """
        line = re.sub('^11$', 'eleven', line)
        line = re.sub('^12$', 'twelve', line)
        line = re.sub('^13$', 'thirteen', line)
        line = re.sub('^14$', 'fourteen', line)
        line = re.sub('^15$', 'fifteen', line)
        line = re.sub('^16$', 'sixteen', line)
        line = re.sub('^17$', 'seventeen', line)
        line = re.sub('^18$', 'eighteen', line)
        line = re.sub('^19$', 'nineteen', line)
        line = re.sub('^20$', 'twenty', line)
        line = re.sub('^10$', 'ten', line)
        line = re.sub('^0$', 'zero', line)
        line = re.sub('^1$', 'one', line)
        line = re.sub('^2$', 'two', line)
        line = re.sub('^3$', 'three', line)
        line = re.sub('^4$', 'four', line)
        line = re.sub('^5$', 'five', line)
        line = re.sub('^6$', 'six', line)
        line = re.sub('^7$', 'seven', line)
        line = re.sub('^8$', 'eight', line)
        line = re.sub('^9$', 'nine', line)
        return line

    def whMovement(self, root):
        """Performs WH - movement on a tree.
        """
        stack = [[]]  # A hack for closure support
        found = [False]

        def traverseFindTopClass(node, className):
            if not found[0]:
                stack[0].append(node)
                if node.className == className:
                    found[0] = True
                else:
                    for child in node.children:
                        traverseFindTopClass(child, className)
                    if not found[0]:
                        del stack[0][-1]

        # Find the subject (first NP) and change determiner to 'the'
        traverseFindTopClass(root, 'NP')
        topNoun = None
        if found[0]:
            np = stack[0][-1]
            while np.className != 'DT' and len(np.children) > 0:
                np = np.children[0]
            if np.className == 'DT' and np.text.lower() == 'a':
                np.text = 'the'
            np = stack[0][-1]

            def lookForNoun(np):
                if len(np.children) > 0:
                    for child in np.children:
                        answer = lookForNoun(child)
                        if (answer != None):
                            return answer
                    return None
                else:
                    if np.className == 'NN' or np.className == 'NNS':
                        return np
                    else:
                        return None
            topNoun = lookForNoun(np)

        # Find the top verb
        found[0] = False
        stack[0] = []
        traverseFindTopClass(root, 'VP')
        topVP = None
        if found[0]:
            topVP = stack[0][-1]

        # First look for the position of WHNP
        found[0] = False
        stack[0] = []
        traverseFindTopClass(root, 'WHNP')
        if not found[0]:
            return False

        # Check if the WHNP is inside an SBAR, not handling this case for now.
        insideSBar = False
        # Check if inside NP, violates A-over-A principal
        insideNP = False
        insideVP = False

        whStack = stack[0][:]
        whPosition = len(whStack) - 1

        for item in whStack:
            if item.className == 'SBAR':
                insideSBar = True
            elif item.className == 'NP' and item.level > 1:
                insideNP = True
            elif insideNP and item.className == 'VP':
                insideVP = True

        # Look for VP
        found[0] = False
        stack[0] = []
        traverseFindTopClass(root, 'VP')

        node = root
        parent = root
        while len(node.children) > 0:
            parent = node
            node = node.children[0]

        if parent.className == 'WHNP':
            if found[0]:
                # Add in missing verbs if possible
                vpnode = stack[0][-1]
                vpchild = vpnode.children[0]
                frontWord = None
                if vpchild.className == 'VBG':  # only doing present, no is/are
                    verb = 'are' if root.answer.className == 'NNS' else 'is'
                    verbnode = TreeNode('VB', verb, [], vpchild.level)
                    vpnode.children.insert(0, verbnode)
            return True

        if insideSBar:
            return False
        if insideVP:
            return False

        if not found[0]:
            return False

        # Look for the verb that needs to be moved to the front.
        vpnode = stack[0][-1]
        vpchild = vpnode.children[0]
        frontWord = None
        if vpchild.className == 'VBZ':  # is, has, singular present
            if vpchild.text == 'is':
                frontWord = vpchild
                vpnode.children.remove(vpchild)
            elif vpchild.text == 'has':  # Could be has something or has done
                done = False
                for child in vpnode.children:
                    if child.className == 'VP':
                        done = True
                        break
                if done:
                    frontWord = vpchild
                    vpnode.children.remove(vpchild)
                else:
                    frontWord = TreeNode('VBZ', 'does', [], 0)
                    vpchild.text = 'have'
                    vpchild.className = 'VB'
            else:
                # need to lemmatize the verb and separate does
                frontWord = TreeNode('VBZ', 'does', [], 0)
                vpchild.className = 'VB'
                vpchild.text = lemmatizer.lemmatize(vpchild.text, 'v')
            pass
        elif vpchild.className == 'VBP':  # do, have, present
            if vpchild.text == 'are':
                frontWord = vpchild
                vpnode.children.remove(vpchild)
            else:
                frontWord = TreeNode('VBP', 'do', [], 0)
                vpchild.className = 'VB'
            pass
        elif vpchild.className == 'VBD':  # did, past tense
            if vpchild.text == 'was' or vpchild.text == 'were':
                frontWord = vpchild
                vpnode.children.remove(vpchild)
            elif vpchild.text == 'had':  # Could be had something or had done
                done = False
                for child in vpnode.children:
                    if child.className == 'VP':
                        done = True
                        break
                if done:
                    frontWord = vpchild
                    vpnode.children.remove(vpchild)
                else:
                    frontWord = TreeNode('VBD', 'did', [], 0)
                    vpchild.text = 'have'
                    vpchild.className = 'VB'
            else:
                # need to lemmatize the verb and separate did
                frontWord = TreeNode('VBD', 'did', [], 0)
                vpchild.className = 'VB'
                vpchild.text = lemmatizer.lemmatize(vpchild.text, 'v')
            pass
        elif vpchild.className == 'MD':  # will, may, shall
            frontWord = vpchild
            vpnode.children.remove(vpchild)
            pass
        elif vpchild.className == 'VBG':  # only doing present, no is/are
            verb = 'are' if topNoun != None and topNoun.className == 'NNS' else 'is'
            frontWord = TreeNode('VBZ', verb, [], 0)

        # Verb not found
        if frontWord is None:
            return False

        # Remove WHNP from its parent.
        whStack[whPosition - 1].children.remove(whStack[whPosition])
        bigS = TreeNode('S', '', [whStack[whPosition], stack[0][1]], 0)
        stack[0][0].children = [bigS]
        bigS.children[1].children.insert(0, frontWord)

        # Reassign levels to the new tree.
        root.relevel(0)
        return True

    def splitCCStructure(self, root):
        """Split composite sentences.
        Find(ROOT(S ...)(CC ...)(S ...)) structure and split them into
        separate trees.
        Issue: need to resolve coreference in the later sentences.
        """
        roots = []

        # Directly search for the top-most S.
        node = root.children[0]
        if node.className == 'S':
            if len(node.children) >= 3:
                childrenClasses = []
                for child in node.children:
                    childrenClasses.append(child.className)
                renew = True
                index = 0
                for c in childrenClasses:
                    if c == 'S' and renew:
                        root_ = TreeNode('ROOT', '', [node.children[index]], 0)
                        root_.relevel(0)
                        roots.append(root_)
                    elif c == 'CC':
                        renew = True
                    index += 1
        if len(roots) == 0:
            roots.append(root)
        return roots

    def lookupLexname(self, word):
        """Look up lex name of a word in WordNet.
        """
        if word in self.lexnameDict:
            return self.lexnameDict[word]
        else:
            synsets = wordnet.synsets(word)
            # Just pick the first definition
            if len(synsets) > 0:
                self.lexnameDict[word] = synsets[0].lexname()
                return self.lexnameDict[word]
            else:
                return None

    def askWhere(self, root):
        """Ask location type questions.
        """
        found = [False]
        answer = ['']

        def traverse(node, parent):
            # Ask one question for now.
            cont = True

            if node.text.lower() == 'this' or \
                    node.text.lower() == 'that' or \
                    node.text.lower() == 'there':
                node.text = ''

            if len(node.children) > 1 and \
                    node.children[1].className == 'VP':
                c = node.children[1]
                while(len(c.children) > 0):
                    c = c.children[0]
                if c.text.lower() in blackListVerbLocation:
                    cont = False

            if not found[0] and cont and node.className != 'PP':
                for child in node.children:
                    traverse(child, node)
            if node.className == 'PP' and \
                    node.children[0].text == 'in':
                c = node

                while(len(c.children) > 0 and
                      (c.children[-1].className == 'NP'
                       or c.children[-1].className == 'NN')):
                    c = c.children[-1]

                if c.className == 'NN'and \
                        self.lookupLexname(c.text) == 'noun.artifact' and \
                        not c.text.lower() in blackListLocation:
                    found[0] = True
                    answer[0] = c.text
                    # Treat ``where'' as WHNP for now.
                    where = TreeNode('WRB', 'where', [], 0)
                    parent.children.insert(parent.children.index(node),
                                           TreeNode('WHNP', '', [where], 0))
                    parent.children.remove(node)
                    # Remove other PP and ADVP in the parent
                    for child in parent.children:
                        if child.className == 'PP' or \
                                child.className == 'ADVP':
                            parent.children.remove(child)
        traverse(root, None)
        if found[0]:
            if self.whMovement(root):
                if root.children[0].children[-1].className != '.':
                    root.children[0].children.append(TreeNode('.', '?', [], 2))
                return [(root.toSentence().lower(), answer[0])]
            else:
                return []
        else:
            return []

    def askWhoWhat(self, root):
        """Ask object type questions.
        """
        found = [False]  # A hack for closure support in python 2.7
        answer = ['']
        # Unlike in 'how many', here we enumerate all possible 'what's
        rootsReplaceWhat = [[]]

        def traverse(node, parent):
            # if node.className != 'PP':
            cont = True
            # For now, not asking any questions inside PP.
            if node.className == 'PP' and node.text.lower() in blackListPrep:
                cont = False
            if (node.level > 1 and node.className == 'S') or \
                    node.className == 'SBAR':
                # Ignore possible answers in any clauses.
                cont = False
            ccNoun = False
            for child in node.children:
                if child.className == 'CC' or child.className == ',':
                    ccNoun = True
                    break
            if node.className == 'NP' and ccNoun:
                cont = False

            if len(node.children) > 1 and \
                    node.children[1].className == 'PP':
                cont = False

            if len(node.children) > 1 and \
                    node.children[1].className == 'VP':
                c = node.children[1]
                while(len(c.children) > 0):
                    c = c.children[0]
                if c.text.lower() in blackListVerb:
                    cont = False

            if node.className == 'VP' and \
                (node.children[0].text.startswith('attach') or
                 node.children[0].text.startswith('take')):
                cont = False

            # TRUNCATE SBAR!!!!!
            for child in node.children:
                if child.className == 'SBAR' or \
                        (child.level > 1 and child.className == 'S'):
                    node.children.remove(child)

            if cont:
                for child in node.children:
                    if child.className != 'PP' and \
                            child.className != 'ADVP':
                        traverse(child, node)

            if node.className == 'NP' and not ccNoun:
                replace = None
                whword = None
                for child in node.children:
                    # A wide ``angle'' view of the kitchen work area
                    if parent is not None:
                        if node.children.index(child) == len(node.children) - 1:
                            if parent.children.index(node) != \
                                    len(parent.children) - 1:
                                if parent.children[
                                        parent.children.index(node) + 1]\
                                        .className == 'NP':
                                    break
                        # The two people are walking down the ``beach''
                        foundDown = False
                        if parent.children.index(node) != 0:
                            for sib in parent.children[
                                    parent.children.index(node) - 1].children:
                                if sib.text == 'down':
                                    foundDown = True
                        if foundDown:
                            break
                    if child.className == 'NN' or child.className == 'NNS':
                        lexname = self.lookupLexname(child.text)
                        if lexname is not None:
                            if lexname in whiteListLexname and \
                                    not child.text.lower() in blackListNoun:
                                whword = 'what'
                            if whword is not None:
                                answer[0] = child.text
                                found[0] = True
                                replace = child
                if replace != None and not answer[0].lower() in blackListNoun:
                    what = TreeNode('WP', whword, [], node.level + 1)
                    children_bak = copy.copy(node.children)
                    toremove = []

                    for child in node.children:
                        lexname = self.lookupLexname(child.text)
                        if child != replace and (
                                lexname != 'noun.act' or
                                child.className != 'NN' or
                                child.text.lower() in blackListCompoundNoun):
                            toremove.append(child)
                    for item in toremove:
                        node.children.remove(item)
                    if len(node.children) == 1:
                        node.children = [what]
                        node.className = 'WHNP'
                    else:
                        node.children[node.children.index(replace)] = TreeNode(
                            'WHNP', '', [what], node.level + 2)
                    rootcopy = root.copy()
                    rootcopy.answer = replace
                    rootsReplaceWhat[0].append(rootcopy)
                    node.className = 'NP'
                    node.children = children_bak

        rootsSplitCC = self.splitCCStructure(root)
        for r in rootsSplitCC:
            traverse(r, None)
            for r2 in rootsReplaceWhat[0]:
                if r2.children[0].children[-1].className != '.':
                    r2.children[0].children.append(TreeNode('.', '?', [], 2))
                else:
                    r2.children[0].children[-1].text = '?'
                if found[0]:
                    self.whMovement(r2)
                    yield (r2.toSentence().lower(),
                           self.escapeNumber(r2.answer.text.lower()))
                else:
                    pass
            found[0] = False
            answer[0] = None
            rootsReplaceWhat[0] = []

    def askHowMany(self, root):
        """Ask couting questions.
        """
        # A hack for closure support in python 2.7
        found = [False]
        answer = [None]

        def traverse(node):
            if not found[0]:
                ccNoun = False
                cont = True
                for child in node.children:
                    if child.className == 'CC' or child.className == ',':
                        ccNoun = True
                        break

                if node.className == 'NP' and ccNoun:
                    cont = False

                if node.className == 'PP':
                    cont = False

                if cont:
                    for child in node.children:
                        traverse(child)
                    if node.className == 'NP' and (
                            node.children[-1].className == 'NNS' or
                            node.children[-1].className == 'NN') and \
                            not node.children[-1].text.startswith('end'):
                        count = None
                        for child in node.children:
                            if child.className == 'CD':
                                if not child.text.lower() in \
                                        blackListNumberNoun:
                                    found[0] = True
                                    answer[0] = child
                                    count = child
                        if found[0] and count is not None:
                            how = TreeNode('WRB', 'how', [], node.level + 2)
                            many = TreeNode('JJ', 'many', [], node.level + 2)
                            howmany = TreeNode('WHNP', '', [how, many],
                                               node.level + 1)
                            children = [howmany]
                            children.extend(node.children[
                                node.children.index(count) + 1:])
                            node.children = children
                            node.className = 'WHNP'
                            return

        roots = self.splitCCStructure(root)

        for r in roots:
            traverse(r)
            if r.children[0].children[-1].className != '.':
                r.children[0].children.append(TreeNode('.', '?', [], 2))
            else:
                r.children[0].children[-1].text = '?'
            if found[0] and \
                    not answer[0].text.lower() in blackListNumberNoun:
                r.answer = answer[0]
                self.whMovement(r)
                yield (r.toSentence().lower(), self.escapeNumber(
                    answer[0].text.lower()))
            found[0] = False
            answer[0] = None

    def askColor(self, root):
        """Ask color questions.
        """
        found = [False]
        answer = [None]
        obj = [None]
        qa = [[]]
        template = 'what is the color of the %s ?'

        def traverse(node):
            for child in node.children:
                traverse(child)
            if node.className == 'NP':
                for child in node.children:
                    if child.className == 'JJ' and \
                            child.text.lower() in whiteListColorAdj:
                        found[0] = True
                        answer[0] = child
                    if child.className == 'CC' and \
                            child.text == 'and':
                        # Blue and white? No.
                        found[0] = False
                        answer[0] = None
                        break
                    if (child.className == 'NN' or
                            child.className == 'NNS') and \
                            not child.text.lower() in blackListColorNoun:
                        obj[0] = child
                if found[0] and obj[0] is not None:
                    qa[0].append(((template % obj[0].text).lower(),
                                  answer[0].text.lower()))
                    found[0] = False
                    obj[0] = None
                    answer[0] = None
        traverse(root)
        return qa[0]


def lookupSynonym(word):
    """Lookup synonyms in the table.
    """
    if word in synonymConvert:
        return synonymConvert[word]
    else:
        return word


def questionGen(parseFilename, outputFilename=None):
    """Generates questions.
    """
    startTime = time.time()
    qCount = 0
    numSentences = 0
    parser = TreeParser()
    gen = QuestionGenerator()
    questionAll = []

    def newTree():
        return parser.rootsList[0].copy()

    def addQuestion(sentId, origSent, question, answer, typ):
        questionAll.append((sentId, origSent, question, answer, typ))

    def addItem(qaitem, origSent, typ):
        ques = qaitem[0]
        ans = lookupSynonym(qaitem[1])
        log.info('Question {:d}: {} Answer: {}'.format(
            qCount, ques, ans))
        addQuestion(numSentences, origSent, ques, ans, typ)

    with open(parseFilename) as f:
        for line in f:
            if len(parser.rootsList) > 0:
                origSent = parser.rootsList[0].toSentence()

                # 0 is what-who question type
                for qaitem in gen.askWhoWhat(newTree()):
                    # Ignore too short questions
                    if len(qaitem[0].split(' ')) < 5:
                        continue
                    qCount += 1
                    addItem(qaitem, origSent, 0)

                # 1 is how-many question type
                for qaitem in gen.askHowMany(newTree()):
                    qCount += 1
                    addItem(qaitem, origSent, 1)

                # 2 is color question type
                for qaitem in gen.askColor(newTree()):
                    qCount += 1
                    addItem(qaitem, origSent, 2)

                # 3 is location question type
                for qaitem in gen.askWhere(newTree()):
                    qCount += 1
                    addItem(qaitem, origSent, 3)

                del(parser.rootsList[0])
                numSentences += 1
            parser.parse(line)

    log.info('Number of sentences: {:d}'.format(numSentences))
    log.info('Time elapsed: {:f} seconds'.format(time.time() - startTime))
    log.info('Number of questions: {:d}'.format(qCount))

    if outputFilename is not None:
        log.info('Writing to output {}'.format(
            os.path.abspath(outputFilename)))
        with open(outputFilename, 'wb') as f:
            pkl.dump(questionAll, f)

    pass


def printQAs(qaiter, qid=0):
    """Print QA pair.
    """
    for qaitem in qaiter:
        log.info('Question {:d}: {} Answer: {}'.format(
            qid, qaitem[0], qaitem[1]))

    pass


def stanfordParseSingle(parserFolder, sentence):
    """Call stanford parser on a single sentence.
    """
    tmpFname = 'tmp.txt'
    tmpOutFname = 'tmpout.txt'
    with open(tmpFname, 'w') as f:
        f.write(sentence)
    stanfordParseFile(parserFolder, tmpFname, tmpOutFname)
    with open(tmpOutFname) as f:
        result = f.read()
    os.remove(tmpFname)
    os.remove(tmpOutFname)

    return result


def stanfordParseFile(parserFolder, inputFilename, outputFilename):
    """Call stanford parser on an input file.
    """
    stanfordParserPath = os.path.join(parserFolder, 'lexparser.sh')

    with open(outputFilename, 'w') as fout:
        subprocess.call([stanfordParserPath, inputFilename], stdout=fout)

    pass


def runSentence(parserFolder, sentence):
    """Run a single sentence.
    """
    s = stanfordParseSingle(parserFolder, sentence)
    s = s.split('\n')
    parser = TreeParser()
    gen = QuestionGenerator()
    for i in range(len(s)):
        parser.parse(s[i] + '\n')
    tree = parser.rootsList[0]
    log.info('Parser result:')
    log.info(tree)
    qaiter = gen.askWhoWhat(tree.copy())
    printQAs(qaiter)
    qaiter = gen.askHowMany(tree.copy())
    printQAs(qaiter)
    qaiter = gen.askColor(tree.copy())
    printQAs(qaiter)
    qaiter = gen.askWhere(tree.copy())
    printQAs(qaiter)

    pass


def runList(parserFolder, inputFilename, outputFilename=None):
    parseFilename = inputFilename + '.parse.txt'
    stanfordParseFile(parserFolder, inputFilename, parseFilename)
    questionGen(parseFilename, outputFilename)

    pass


def parseArgs():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Question Generator')
    parser.add_argument(
        '-parser_path',
        default='/home/mren/third_party/stanford-parser-full-2015-04-20',
        help='Path to stanford parser')
    parser.add_argument(
        '-sentence', default=None, help='Single sentence input')
    parser.add_argument(
        '-list', default=None, help='List file input')
    parser.add_argument(
        '-parsed_file', default=None, help='Parsed file input')
    parser.add_argument(
        '-output', default=None, help='Output file name')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parseArgs()
    if args.sentence:
        log.info('Single sentence mode')
        runSentence(parserFolder=args.parser_path, sentence=args.sentence)
    elif args.list:
        log.info('List of sentences mode')
        runList(parserFolder=args.parser_path,
                inputFilename=args.list, outputFilename=args.output)
    elif args.parsed_file:
        log.info('Pre-parsed file mode')
        questionGen(parseFilename=args.parsed_file, outputFilename=args.output)
    else:
        raise Exception(
            ('You must provide one of the three options: -sentence, -list, ',
             'or -parsed_file'))
