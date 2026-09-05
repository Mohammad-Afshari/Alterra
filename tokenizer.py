import re
from collections import Counter, defaultdict
from tqdm import tqdm

class SubwordTokenizer:
    """This is a  subword tokenizer."""

    def __init__(self, regex_rule=None , special_tokens=None, basic_tokens=None):
        self.is_trained = False
        self.basic_tokens = basic_tokens or list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz123456789.,;:!?|')
        self.regex_rule = regex_rule or re.compile(r'[^a-z123456789\!\?\.\,><]+')
        # self.regex_rule = regex_rule or re.compile(r'[^a-z123456789\.,!\?;\:\-\(\)\[\]\{\}\+\/\\@#\$%\^&\*><]+')
        # r'[^a-z0-9><./,:-]+'

        self.freq_vocab= None
        self.merge_rules = None
        self.token_to_id = None
        self.id_to_token = None

        self.WORD_END = '</w>'

        self.special_tokens = special_tokens or ['<PAD>', '<UNK>']

        self.punctation_tokens = {
            '"': '<DOUBLE_COT>',
            "'": '<SINGLE_COT>'
        }
        for punc_tok in self.punctation_tokens.values():
            self.special_tokens.append(punc_tok)
        

    # ---------------------------------------------------

    def normalize(self, text):
        for punc_tok in self.punctation_tokens.keys():
            text = text.replace(punc_tok, f' {self.punctation_tokens[punc_tok]} ')

        # text = re.sub(self.regex_rule, ' ', text)
        # text = re.sub(r'\s+', ' ', text)
        # return text.strip()
        return text
    
    # ---------------------------------------------------


    def is_special_token(self, word):
        if word in self.special_tokens:
            return True
        else:
            return False

    # ---------------------------------------------------

    def create_freq_vocab(self, text):
        words = text.split()
        words_frequency_vocab = Counter(words)

        freq_vocab = {}
        for word, freq in words_frequency_vocab.items():
            if self.is_special_token(word):
                freq_vocab[(word,)] = freq
            else:
                letters = list(word)
                letters.append(self.WORD_END)
                freq_vocab[tuple(letters)] = freq

            self.freq_vocab = freq_vocab
    
    # ---------------------------------------------------

    def get_pairs_freq(self):
        """Returns every pair of each word in freq_vocab and its frequency."""
        pairs = defaultdict(int)
        for word, freq in self.freq_vocab.items():
            letters = word
            for i in range(len(letters)-1):
                pairs[letters[i], letters[i+1]] += freq

        return pairs
    
    # ---------------------------------------------------
    
    def merge_vocab(self, pair):
        """merges the given pair(here:most frequent pair) with every word in freq_vocab (if pair exist in freq_vocab words)
        and returns a new freq_vocab that its words contain pairs instead letters"""

        new_freq_vocab = {} # new vocab that shows the frequency of each word that contains most used pairs
        pair_text = ''.join(pair)

        for word, freq in self.freq_vocab.items():
            letters = list(word)

            i = 0
            new_word = [] # for every word in freq_vocab, the word merges with the most frequnce pairs if the pair exist in the word

            while i < len(letters):
                if i < len(letters)-1 and (letters[i], letters[i+1]) == pair:
                    new_word.append(pair_text)
                    i += 2
                else:
                    new_word.append(letters[i])
                    i += 1
            new_freq_vocab[tuple(new_word)] = freq
        return new_freq_vocab

    # ---------------------------------------------------

    def create_merge_rules(self, num_merges=None):
        """It is generally not advisable to continue with the least frequent pair merging repeatedly.
        Therefore, you should avoid setting a very high number of merges and specifying an excessively large value for num_merge_rules.
        If you have no idea how to set the num_merges, set num_merges to zero ; the code will intelligently determine the required num_merges on its own
        \nNOTE: THIS FUNCTION CAN BE USED FOR ONLY ONE TIMES. THEREFORE TOKENIZER CAN BE TRAINED FOR ONLY ONE TIMES"""


        if num_merges == 0:
            import statistics
            num_merges = int(statistics.mean(list(self.freq_vocab.values())) * 100)
            # num_merges = int(statistics.mode(list(self.freq_vocab.values())))
            # num_merges = int(statistics.median(list(self.freq_vocab.values())))
            print(f'num_merges will be {num_merges}')

        merge_rules = []

        print('>> Merging pairs with frequncy vocab...')
        for i in tqdm(range(num_merges)):
            pairs = self.get_pairs_freq()
            if not pairs:
                print('>> There is no pairs left.')
                break
            most_frequent_pair = max(pairs, key=pairs.get) # the most frequent pair in the pairs of all words
            self.freq_vocab = self.merge_vocab(most_frequent_pair)
            merge_rules.append(most_frequent_pair)

        if len(merge_rules) != 0:
            self.merge_rules = merge_rules

        print(f'>> Merging is over. total merge rules:{len(self.merge_rules)}')

    # ---------------------------------------------------

    def build_token_id_mapping(self):
        vocab = []
        vocab.extend(self.special_tokens)
        vocab.extend(self.basic_tokens)

        for pair in self.merge_rules:
            vocab.append(''.join(pair))

        token_to_id = {}
        id_to_token = {}
        for id, tok in enumerate(sorted(set(vocab))):
            token_to_id[tok] = id
            id_to_token[id] = tok

        self.token_to_id = token_to_id
        self.id_to_token = id_to_token


    # ---------------------------------------------------
    
    def get_vocab_size(self):
        if self.is_trained:
            return len(list(self.id_to_token.values()))
        else:
            return 0


    # ---------------------------------------------------
    
    def fit(self, text_to_fit, num_merge_rules):
        if not self.is_trained:
            text_to_fit = self.normalize(text_to_fit)
            self.create_freq_vocab(text_to_fit)
            self.create_merge_rules(num_merge_rules)
            self.build_token_id_mapping()

            self.is_trained = True
            
        else:
            raise ValueError(
                "Tokenizer has already been trained! "
                "You cannot retrain a tokenizer that is already fitted. "
                "Please create a new instance of the tokenizer or reset it using reset() function."
            )

    # ---------------------------------------------------

    def reset(self):
        self.special_tokens = ['<PAD>', '<UNK>', '</w>']
        self.basic_tokens = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz123456789.,;:')
        self.regex_rule = re.compile(r'[^a-z123456789\!\?\.\,><]+')
        
        self.freq_vocab= None
        self.merge_rules = None
        self.token_to_id = None
        self.id_to_token = None

        self.WORD_END = '</w>'
        self.is_trained = False

    # ---------------------------------------------------

    def tokenize_word(self, word) -> list:
        """Split a word into subword token strings (always a list)."""
        letters = list(word)
        letters.append(self.WORD_END)

        current_tokens = letters

        for pair in self.merge_rules:
            i=0

            new_tokens = []

            while i < len(current_tokens):
                if ( i+1 < len(current_tokens) ) and ( pair == (current_tokens[i], current_tokens[i+1]) ):
                    new_tokens.append("".join(pair))
                    i+=2
                else:
                    new_tokens.append(current_tokens[i])
                    i+=1

            current_tokens = new_tokens

        return current_tokens
        
    # ---------------------------------------------------
    
    def tokenize_text(self, text):
        """Split text into subword token strings (always a list)."""
        if not text:
            return []

        tokens = []
        words = text.split()
        for word in words:
            if self.is_special_token(word):
                tokens.append(word)
            else:
                tokens.extend(self.tokenize_word(word))

        return tokens
        

    # ---------------------------------------------------

    def encode(self, text):
        tokens = self.tokenize_text(text)

        token_ids = []

        for token in tokens:
            if str(token) in self.token_to_id or token in self.special_tokens:
                token_ids.append( self.token_to_id[str(token)] )
            else: 
                token_ids.append( self.token_to_id['<UNK>'] )

        return token_ids

    # ---------------------------------------------------

    def decode(self, id_array) -> str:
        tokens = []
        for id in id_array:
            tokens.append( self.id_to_token[id] )

        tokens = "".join(tokens)
        tokens = tokens.replace('</w>', ' ')
        for punc_tok in self.punctation_tokens.keys():
            tokens = tokens.replace(self.punctation_tokens[punc_tok], punc_tok)

        return tokens

    # ---------------------------------------------------

    def save_configs(self, path):
        import json

        configs = {
            "special_tokens": self.special_tokens,
            "basic_tokens": self.basic_tokens,
            "merge_rules": [list(pair) for pair in self.merge_rules],   # convert tuple -> list
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},  # convert keys -> string
            "regex_rule": self.regex_rule.pattern,
            "is_trained": str(self.is_trained),
            "WORD_END": self.WORD_END
        }

        with open(f"{path}/tokenizer_configs.json", 'w', encoding='utf-8') as f:
            json.dump(configs, f, ensure_ascii=False, indent=4)


    # ---------------------------------------------------

    def load_configs(self, config_file_path):
        import json
        with open(config_file_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)

        self.special_tokens = configs["special_tokens"]
        self.basic_tokens = configs["basic_tokens"]

        # convert lists → tuples
        self.merge_rules = [tuple(pair) for pair in configs["merge_rules"]]

        # restore int keys
        self.id_to_token = {int(k): v for k, v in configs["id_to_token"].items()}
        self.token_to_id = configs["token_to_id"]

        self.is_trained = bool(configs["is_trained"])
        self.WORD_ENDc = configs["WORD_END"]
        # restore regex
        import re
        self.regex_rule = re.compile(configs["regex_rule"])