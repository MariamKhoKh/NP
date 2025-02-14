# It's sceleton for puzzle creating and searching in python
# Btw 1 hour it's little time for done it, I've implemented main features
#   - creating puzzle by X and Y including consistently words (and reverse) without diagonal
#   - creating searching for consistently and reverse words
#   - it may be implemented additional functions for each object for implement not done logic
# It's just a quick simple representation without unit tests and tox
# Example usage: python puzzly.py -y 8 -x 8
# Exampel usage 2: python puzzly.py
# Fully tested on Python 2.7.11
# UPD: words.txt may be located in the same directory as script

import os
import click
import random

# Constants for word path and letters
WORDS_LIST = 'words.txt'
LETTER_LIST = list('abcdefghijklmnopqrstuvwxyz')


class WordDictionary(object):
    """
    Represents Words Dictionary object
    """

    def __init__(self):
        self.words = []

    def open_file(self):
        if os.path.exists(WORDS_LIST):
            self.words = open(WORDS_LIST, 'r').read().split('\n')
        else:
            raise Exception('Word list in path {0} not found!'.format(WORDS_LIST))


class Board(object):
    """
    Represents Board object
    :arg x - multiplier by X-axis
    :arg y - multiplier Y-axis
    :arg words_list - must be WordDictionary obj
    """

    def __init__(self, x, y, words_list):
        self.x = x
        self.y = y
        self.words_list = words_list
        self.board = []

    def get_word(self):
        """
        Function get random word from Words Dictionary
        Also randomly reverts if True
        :return: string word
        """

        word = random.choice(self.words_list)
        while len(word) > self.x - 2:
            word = random.choice(self.words_list)
        is_reverse = random.randint(0, 1)
        if is_reverse:
            list(word).reverse()
        return word

    def generate(self):
        """
        Function for generating board by x and y
        Not returning, board placed in obj var board
        """

        for y_str in range(self.y):
            word = self.get_word()
            x_str = word.upper()
            if len(word) != self.x:
                for num in range(self.x - len(word)):
                    cnt = random.randint(0, 1)
                    if cnt == 1:
                        x_str = random.choice(LETTER_LIST) + x_str
                    else:
                        x_str = x_str + random.choice(LETTER_LIST)
            self.board.append(list(x_str))


class Finder(object):
    """
    Respresents find solution for puzzle board
    :arg board - must be Board.board (list of lists)
    :arg word_list - bust be WordDictionary.word
    """

    def __init__(self, board, word_list):
        self.board = board
        self.word_list = word_list

    def find_normal(self):
        """
        Function that make searcher in simple normal way
        (consistently)
        """
        for x_str in self.board:
            x_str_conv = ''.join(x_str)
            for word in self.word_list:
                up = word.upper()
                if up in str(x_str_conv) and up:
                    print ("Found word", up)

    def find_reverse(self):
        """
        Function that make searcher in simple normal way
        (consistently)
        """
        for x_str in self.board:
            x_str.reverse()
            x_str_conv = ''.join(x_str)
            for word in self.word_list:
                up = word.upper()
                if up in str(x_str_conv) and up:
                    print("Found reverse word", up)


@click.command()
@click.option('-x', default=15, help='size for generating by x')
@click.option('-y', default=15, help='size for generating by y')
def main(x, y):
    """
    Main function for calling all logic
    :param x: multiplier for board by X-axis
    :param y: multiplier for board by X-axis
    """

    # Read words.txt
    wd = WordDictionary()
    wd.open_file()
    print ('Found {0} words in {1}'.format(len(wd.words), WORDS_LIST))

    # Create puzzle
    b = Board(x, y, wd.words)
    b.generate()

    # Print generated puzzle
    for y_str in b.board:
        print( '[%s]' % ' '.join(map(str, y_str)))

    # Trying to find hidden word
    find = Finder(b.board, wd.words)
    find.find_normal()
    find.find_reverse()

    print ('Thanks for using!')

main()