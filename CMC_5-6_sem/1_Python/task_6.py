def check(s: str, filename: str):
    words = s.lower().split()
    word_cnt = {}
    
    for word in words:
        if word in word_cnt:
            word_cnt[word] += 1
        else:
            word_cnt[word] = 1
    
    sorted_words = sorted(word_cnt.keys())
    
    with open(filename, 'w') as file:
        for word in sorted_words:
            file.write(f"{word} {word_cnt[word]}\n")
