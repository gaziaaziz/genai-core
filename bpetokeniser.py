corpus = ["low", "lower", "lowest", "newest", "widest"]
def get_symbol_corpus(corpus):
    symbol_corpus = []
    for word in corpus:
        char_list = list(word)
        char_list.append('</w>')
        symbol_corpus.append(char_list)
    return symbol_corpus

def get_vocab(symbol_corpus):
    vocab = set()
    for li in symbol_corpus:
        for symbol in li:
            vocab.add(symbol)
    return vocab

def get_pair_freq(symbol_corpus):
    pair_freq = {}
    for symbols in symbol_corpus:
        for i in range(len(symbols)-1):
            pair = (symbols[i],symbols[i+1])
            pair_freq[pair] = pair_freq.get(pair, 0)+1
    return pair_freq
    
def get_best_pair(pair_freq):
    pair_freq = sorted(pair_freq.items(), key = lambda x: x[1], reverse=True)
    best_pair = pair_freq[0][0]
    return best_pair

def merge_pair(symbol_corpus, best_pair):
    new_symbol_corpus = []
    for symbols in symbol_corpus:
        new_symbols_list = []
        i = 0
        while i < len(symbols):
            if i < (len(symbols)-1) and (symbols[i], symbols[i+1]) == best_pair:
                new_symbols_list.append(best_pair[0]+best_pair[1])
                i += 2
            else:
                new_symbols_list.append(symbols[i])
                i += 1
        new_symbol_corpus.append(new_symbols_list)
    return new_symbol_corpus

def get_token_ids(vocab):
    vocab_list = sorted(vocab)
    token_to_id = {}
    id_to_token = {}
    for i,token in enumerate(vocab_list):
        token_to_id[token] = i
        id_to_token[i] = token
    
    return token_to_id, id_to_token

def encode(word, merges, token_to_id):
    new_list = list(word)
    new_list.append('</w>')
    
    for best_pair in merges:
        new_symbol_corpus = merge_pair([new_list], best_pair)
        new_list = new_symbol_corpus[0]
    
    token_ids = []
    
    for token in new_list:
        token_ids.append(token_to_id[token])
    
    return token_ids





vocab_size = 50
symbol_corpus = get_symbol_corpus(corpus)
vocab = get_vocab(symbol_corpus)
num_merges = max(0, vocab_size - len(vocab))
merges = []
for i in range(num_merges):
    pair_freq = get_pair_freq(symbol_corpus)
    if not pair_freq:
        break
    best_pair = get_best_pair(pair_freq)
    merges.append(best_pair)
    vocab.add(best_pair[0]+best_pair[1])
    symbol_corpus = merge_pair(symbol_corpus, best_pair)
token_to_id, id_to_token = get_token_ids(vocab)
encoded = encode("lowest", merges, token_to_id)


    
    