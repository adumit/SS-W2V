import pickle
f_nums = "20000_30000"
input_file = "../Data/tagged.en/englishEtiquetado_" + f_nums

num_sentences = 0
with open(input_file, encoding="utf-8", errors="ignore") as f:
    sentences_and_pos = []
    # Currect sentence is a tuple of (words, POS tags)
    cur_sentence = ([], [])
    for line in f:
        if line == "\n" or line == "":
            sentences_and_pos.append(cur_sentence)
            cur_sentence = ([], [])
            num_sentences += 1
        elif line[0] in ["<", ",", ".", ",", ";", ":", "-",
                         "/", "\\", "|", "(", "+", ")", '"'] or len(line) == 1:
            continue
        else:
            split_line = line.lower().split(" ")
            cur_sentence[0].append(split_line[0])
            cur_sentence[1].append(split_line[2])

with open("../Data/Pickled_Sentences/File_" + f_nums, "wb+") as f:
    pickle.dump(obj=sentences_and_pos, file=f)