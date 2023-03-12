import fasttext

model_file = "/workspace/datasets/fasttext/title_model_100.bin"
word_file =  "/workspace/datasets/fasttext/top_words.txt"
synonym_file = "/workspace/datasets/fasttext/synonyms.csv"
threshold = 0.75

model  = fasttext.load_model(model_file)

with open(word_file, "r") as data:
    with open(synonym_file, "w") as output_file:
        lines = data.readlines()
        for line in lines:
            line = line.rstrip('\n')
            synonyms  = model.get_nearest_neighbors(line)
            output_list = [line]
            for synonym in synonyms:
                score = synonym[0]
                word = synonym[1]
                if word!=line and score >= threshold:
                    output_list.append(word)
            if len(output_list) > 1:
                output_line = ','.join(output_list) + "\n"
                output_file.write(output_line)

