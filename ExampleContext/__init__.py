from collections import defaultdict
import ndjson


pairs = ["monastery", "convent",
         "spokesman", "spokeswoman",
         "Catholic_priest", "nun",
         "Dad", "Mom",
         "Men", "Women",
         "councilman", "councilwoman",
         "grandpa", "grandma",
         "grandsons", "granddaughters",
         "prostate_cancer", "ovarian_cancer",
         "testosterone", "estrogen",
         "uncle", "aunt",
         "wives", "husbands",
         "Father", "Mother",
         "Grandpa", "Grandma",
         "He", "She",
         "boy", "girl",
         "boys", "girls",
         "brother", "sister",
         "brothers", "sisters",
         "businessman", "businesswoman",
         "chairman", "chairwoman",
         "colt", "filly",
         "congressman", "congresswoman",
         "dad", "mom",
         "dads", "moms",
         "dudes", "gals",
         "ex_girlfriend", "ex_boyfriend",
         "father", "mother",
         "fatherhood", "motherhood",
         "fathers", "mothers",
         "fella", "granny",
         "fraternity", "sorority",
         "gelding", "mare",
         "gentleman", "lady",
         "gentlemen", "ladies",
         "grandfather", "grandmother",
         "grandson", "granddaughter",
         "he", "she",
         "himself", "herself",
         "his", "her",
         "king", "queen",
         "kings", "queens",
         "male", "female",
         "males", "females",
         "man", "woman",
         "men", "women",
         "nephew", "niece",
         "prince", "princess",
         "schoolboy", "schoolgirl",
         "son", "daughter",
         "sons", "daughters",
         "twin_brother", "twin_sister"]


class ExampleContext(object):
    """instantiate this class on a corpus, then call
        get_example() to get an example.
    """

    def __init__(self, ndjson_file, example_length=50, num_examples=3):
        ''' corpus_file should be the pass to a ndjson file
            example_length should be the length of the examples returned
            can be updated at any time
        '''
        super(ExampleContext, self).__init__()

        self._example_length = None
        self.example_length = example_length

        self._num_examples = None
        self.num_examples = num_examples

        print('Loading corpus from {}'.format(ndjson_file))
        with open(ndjson_file) as f:
            data = ndjson.load(f)

        print("\tLoaded {} sentences.".format(len(data)))

        self.vocab = defaultdict(list)
        for sentence in data:
            words_added = set()
            for word in sentence:
                if word not in words_added:
                    self.vocab[word].append(sentence)
                    words_added.add(word)

        print("\tComputed vocab of {} words.".format(len(self.vocab)))

    @property
    def example_length(self):
        return self._example_length

    @example_length.setter
    def example_length(self, val):
        if not isinstance(val, int):
            raise ValueError("{} must be an int".format(val))

        # make sure example_length is odd
        if val % 2 == 0:
            val += 1
        self._example_length = val

    @property
    def num_examples(self):
        return self._num_examples

    @num_examples.setter
    def num_examples(self, val):
        if not isinstance(val, int):
            raise ValueError("{} must be an int".format(val))

        self._num_examples = val

    def get_examples(self, focus_word, other_words=set(), focus_bonus=2):
        ''' returns a list of sentences containing the most occurances of focus_word.
            if the focus_word does not occur in the corpus, returns an empty list
            if the optional other_words set is provided, returns the sentence
            containing the greatest number of other words.

            The 'best' sentence is computed by summing the occurances of other_words with
            focus_bonus * the occurances of focus_word

            if the resulting example sentence is longer than example_length, the returned
            example will be centered around the focus word and "clipped" at length
            example_length.

            sentences returned as a list of tuples of (token, None or 'focus' or 'other')
        '''
        if focus_word not in self.vocab:
            return []

        possible_sentences = self.vocab[focus_word]
        scored_possible_sentences = []
        for sentence in possible_sentences:
            score = 0
            for word in sentence:
                if word == focus_word:
                    score += focus_bonus
                elif word in other_words:
                    score += 1
            scored_possible_sentences.append((sentence, score))

        sentences = sorted(scored_possible_sentences,
                           key=lambda x: x[1], reverse=True)
        sentences = sentences[:self.num_examples] if len(
            sentences) > self.num_examples else sentences
        sentences = map(lambda x: x[0], sentences)

        def trimmer(sentence):
            if len(sentence) > self.example_length:
                pos_of_focus = sentence.index(focus_word)

                if pos_of_focus < int(self.example_length/2):
                    # focus_word is at front of sentence
                    sentence = sentence[:self.example_length]
                else:
                    padding = int(self.example_length/2)
                    front = pos_of_focus - padding
                    back = pos_of_focus + padding
                    sentence = sentence[front:back+1]
            return sentence

        def tagger(word):
            tag = None
            if word == focus_word:
                tag = 'focus'
            elif word in other_words:
                tag = 'other'
            return (word, tag)

        def sentence_tagger(sentence):
            return list(map(tagger, sentence))

        sentences = list(map(trimmer, sentences))
        sentences = list(map(sentence_tagger, sentences))

        return sentences

    @classmethod
    def pretty_print(cls, l):
        ''' pretty print a list of (word, tag) tuples '''
        words = ''
        tags = ''
        for word, tag in l:
            if tag is None:
                tag = ''
            width = max((len(word), len(tag)))
            paddstr = "{:<" + str(width) + "} "

            words += paddstr.format(word)
            tags += paddstr.format(tag)

        print(words)
        print(tags)


    @classmethod
    def downsample_corpus(cls, corpus_path, output_path, words_to_keep=set(pairs)):
        ''' downsample a corpus by keeping only sentences that contain
            at least one word from the set words_to_keep
        '''
        print("Processing {}...".format(corpus_path))
        print("Looking for sentences containing any of {} words to keep.".format(len(words_to_keep)))
        with open(corpus_path) as corpus, open(output_path, 'w') as output:
            kept = 0
            skipped = 0
            for line in corpus:
                sentence = ndjson.loads(line)[0]
                keep_this_sentence = False
                for word in sentence:
                    if word in words_to_keep:
                        keep_this_sentence = True
                        
                if keep_this_sentence:
                    text = ndjson.dumps([sentence])
                    output.write(text+'\n')
                    kept += 1
                skipped += 1
                

            percent_kept = 100*(float(kept)/(skipped+kept))
            print("kept {}, skipped {} - {:2.4f} percent".format(kept, skipped, percent_kept))


