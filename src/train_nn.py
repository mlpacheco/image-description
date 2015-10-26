def parse_microsoft_sentences():
    pass

def parse_flickr30k_sentences():
    pass

def parse_microsoft_images():
    pass

def parse_flickr30k_images():
    pass

def parse_sentences():
    sentences = []
    parsers = [parse_microsoft_sentences,
               parse_flickr_sentences]
    for p in parsers:
        sentences += p()
    return sentences

def get_sentence_features(sentences):
    pass

def save_sentence_features(filename):
    pass 

def main():
    pass

if __name__ == "__main__"
    main()
