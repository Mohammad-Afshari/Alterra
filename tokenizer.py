import re
from collections import Counter, defaultdict
from tqdm import tqdm
import json

class Tokenizer:
    """A configurable tokenizer supporting BPE and character-level tokenization."""

    def __init__(self, regex_rule:any=None , special_tokens:list=None, allowed_chars:list=None):

        # CONFIGURATIONS
        self.allowed_chars = allowed_chars or list("""ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:()[]{}<>+-=*/!?"'~#%&_$""")
        self.regex_rule = regex_rule or re.compile(f'[^{re.escape( ''.join(self.allowed_chars) )}]')
        self.WORD_END = '</w>'
        self.special_tokens = special_tokens or ['<USER_START>', '<USER_END>', '<AI_START>', '<AI_END>', '<COT_START>', '<COT_END>', '<PAD>', '<UNK>', self.WORD_END]
        self.punctation_tokens = {
            '"': '<DOUBLE_QUOT>',
            "'": '<SINGLE_QUOT>'
        }
        for punc_tok in self.punctation_tokens.values():
            self.special_tokens.append(punc_tok)

        # PARAMETERS
        self.freq_vocab= None
        self.merge_rules = None
        self.token_to_id = None
        self.id_to_token = None
        self.is_trained = False


    # ---------------------------------------------------

    def normalize(self, text:str) -> str:
        """This function cleans and organizes the given text, removes unwanted and unnecessary characters, and delivers it in the requested format."""

        # for punc_tok in self.punctation_tokens.keys():
            # text = text.replace(punc_tok, f' {self.punctation_tokens[punc_tok]} ')

        text = self.regex_rule.sub(' ', text)

        return text
    
    # ---------------------------------------------------


    def is_special_token(self, word:str) -> bool:
        """Indicates whether a token is special, using a boolean value."""

        if word in self.special_tokens:
            return True
        else:
            return False

    # ---------------------------------------------------

    def create_freq_vocab(self, text:str) -> dict:
        """Estimates frequency of each word in the given text and saves it in a dictionary form to Tokenizer.freq_vocab ."""

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

    def get_pairs_freq(self) -> defaultdict:
        """Returns every pair of each word in Tokenizer.freq_vocab and their frequency in a dictionary form."""

        pairs = defaultdict(int)
        for word, freq in self.freq_vocab.items():
            letters = word
            for i in range(len(letters)-1):
                pairs[letters[i], letters[i+1]] += freq

        return pairs
    
    # ---------------------------------------------------
    
    def merge_vocab(self, pair:tuple) -> dict:
        """Merges the given pair(here:most frequent pair) with every word in freq_vocab if pair exist in letters of that word
        and returns a new freq_vocab that it's words contain pairs instead of just letters"""

        new_freq_vocab = {} # new vocab that shows the frequency of each word and it's letters are replaced by pairs
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

    def create_merge_rules(self, num_merges:int) -> None:
        """Merges most frequent pairs with Tokenizer.freq_vocab words for num_merges times."""

        merge_rules = []

        for i in tqdm(range(num_merges)):
            pairs = self.get_pairs_freq()
            if not pairs:
                break

            most_frequent_pair = max(pairs, key=pairs.get) # the most frequent pair in the pairs of all words
            self.freq_vocab = self.merge_vocab(most_frequent_pair)
            merge_rules.append(most_frequent_pair)

        if len(merge_rules) != 0:
            self.merge_rules = merge_rules

    # ---------------------------------------------------

    def build_token_id_mapping(self) -> None:
        """Assigns a unique ID to each token of vocabulary and saves them in dictionary format in Tokenizer.token_to_id & Tokenizer.id_to_token"""

        vocab = []
        vocab += self.special_tokens
        vocab += self.allowed_chars

        if self.merge_rules:
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
    
    def get_vocab_size(self) -> int:
        """Returns the size of vocabulary."""

        if self.is_trained:
            return len(list(self.id_to_token.values()))
        else:
            return len(self.special_tokens + self.allowed_chars)


    # ---------------------------------------------------
    
    def fit(self, text_to_fit:str, num_merge_rules:int) -> None:
        """This function trains the tokenizer with the given text.\n
        Set num_merge_rules to: -1 to let tokenizer decide the num_merges, 0 to set tokenizer to character based mode, or an integer:num_merge_rules to train tokenizer for num_merge_rules times.\n
        NOTE: It is generally not advisable to continue with the least frequent pair merging repeatedly.Therefore, you should avoid setting a very high number of merges and specifying an excessively large value for num_merge_rules.\n
        WARNING: THIS FUNCTION CAN BE USED FOR ONLY ONE TIME; THEREFORE TOKENIZER IS ONE-TIME-TRAINABLE."""

        if not self.is_trained:
            text_to_fit = self.normalize(text_to_fit)

            if not isinstance(num_merge_rules, int):
                raise TypeError("num_merge_rules must be an integer.")

            if num_merge_rules < -1:
                raise ValueError("num_merge_rules mus be -1, 0, or a positive integer.")

            if num_merge_rules == 0:
                self.build_token_id_mapping()

                self.is_trained = True

            elif num_merge_rules == -1:
                import statistics
                num_merges = int(statistics.mean(list(self.freq_vocab.values())) * 100)
                # num_merges = int(statistics.mode(list(self.freq_vocab.values())))
                # num_merges = int(statistics.median(list(self.freq_vocab.values())))
                print(f'num_merges will be {num_merges}')

                self.create_freq_vocab(text_to_fit)
                self.create_merge_rules(num_merges)
                self.build_token_id_mapping()

                self.is_trained = True

            elif num_merge_rules <= 1:
                self.create_freq_vocab(text_to_fit)
                self.create_merge_rules(num_merge_rules)
                self.build_token_id_mapping()

                self.is_trained = True

            
        else:
            raise ValueError(
                "Tokenizer has already been trained! "
                "You cannot retrain a tokenizer that is already fitted. "
                "Please create a new instance of the tokenizer or reset it by calling Tokenizer.reset() function."
            )

    # ---------------------------------------------------

    def reset(self) -> None:
        """Resets tokenizer parameters to default and makes it trainable again (congigurations will not be reset)."""

        self.freq_vocab= None
        self.merge_rules = None
        self.token_to_id = None
        self.id_to_token = None
        self.is_trained = False

    # ---------------------------------------------------

    def tokenize_word(self, word:str) -> list:
        """Breaks the given word into subword token strings"""

        letters = list(word)
        letters.append(self.WORD_END)


        if self.merge_rules:
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

        else:
            return letters
            
        
    # ---------------------------------------------------
    
    def tokenize_text(self, text:str) -> list:
        """Breaks the given text into subword token strings."""

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

    def encode(self, text:str) -> list:
        """Converts(decodes) the given text into token IDs that can be used for input of model."""

        tokens = self.tokenize_text(text)

        token_ids = []

        for token in tokens:
            token_ids.append( self.token_to_id[str(token)] )

        return token_ids

    # ---------------------------------------------------

    def decode(self, token_ids:list) -> str:
        """Converts(encodes) the given token IDs to readable text."""

        tokens = []
        for id in token_ids:
            tokens.append( self.id_to_token[id] )

        tokens = "".join(tokens)
        tokens = tokens.replace('</w>', ' ')
        for punc_tok in self.punctation_tokens.keys():
            tokens = tokens.replace(self.punctation_tokens[punc_tok], punc_tok)

        return tokens

    # ---------------------------------------------------

    def save_configs(self, save_path:str, save_name:str) -> None:
        """Saves the tokenizer configurations and parametes in a json file in the given save path if it is not trained."""

        if not self.is_trained:
            raise ValueError("Tokenizer must be trained first to save it.")
        
        configs = {
            "is_trained": self.is_trained,
            "allowed_chars": self.allowed_chars,
            "regex_rule": self.regex_rule.pattern,

            "merge_rules": [list(pair) for pair in self.merge_rules],   # convert tuple -> list
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},  # convert keys -> string

            "WORD_END": self.WORD_END,
            "special_tokens": self.special_tokens,
            "punctation_tokens": self.punctation_tokens
        }

        with open(f"{save_path}/{save_name}.json", 'w', encoding='utf-8') as f:
            json.dump(configs, f, ensure_ascii=False, indent=4)


    # ---------------------------------------------------

    def load_configs(self, file_path:str) -> None:
        """Loads the tokenizer configurations and parameters from the given json file."""

        with open(file_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)

        self.is_trained = configs["is_trained"]
        self.allowed_chars = configs["allowed_chars"]
        self.regex_rule = re.compile(configs["regex_rule"])

        # convert lists → tuples
        self.merge_rules = [tuple(pair) for pair in configs["merge_rules"]]
        self.token_to_id = configs["token_to_id"]
        self.id_to_token = {int(k): v for k, v in configs["id_to_token"].items()}


        self.WORD_END = configs["WORD_END"]
        self.special_tokens = configs["special_tokens"]
        self.punctation_tokens = configs["punctation_tokens"]