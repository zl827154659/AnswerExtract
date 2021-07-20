from collections import Counter
import pickle
import jieba
import json
import jieba.analyse


def save_freq(flag):
    with open('/home/ray/project/cmrc2018/cmrc2018_' + flag + '.json', 'r') as f:
        dataset = json.load(f)

    questions = []
    word_list = []
    for entry in dataset['data']:
        paragraphs = entry['paragraphs']
        for para in paragraphs:
            context = para['context']
            qas = para['qas']
            for qa in qas:
                question = qa['question']
                questions.append(question)

    for question in questions:
        words = list(jieba.cut(question))
        word_list.extend(words)
    freq = Counter(word_list)
    # del freq['的']
    # del freq['年']
    # del freq['，']
    # del freq['了']
    # del freq['《']
    # del freq['》']
    # del freq['·']
    # del freq[' ']
    # del freq['主要']
    # del freq['中']
    # del freq['和']
    # del freq['与']
    # del freq['后']
    # del freq['-']
    # del freq['以']
    # del freq['上']
    # del freq['会']
    # del freq['分布']
    # del freq['公司']
    # del freq['月']
    # del freq['“']
    # del freq['”']
    # del freq['车站']
    # del freq['开始']
    # del freq['「']
    # del freq['」']
    # del freq['中国']
    # del freq['电影']
    # del freq['地区']
    # del freq['获得']
    # del freq['大']
    # del freq['一个']
    # del freq['作品']
    # del freq['美国']
    # del freq['）']
    # del freq['（']
    # del freq['一般']
    # del freq['所']
    # del freq['叫']
    # del freq['指']
    with open('./bert_for_qa/Counter_' + flag + '.data', 'wb') as f:
        pickle.dump(freq, f)


def load_freq(filename):
    with open(filename, 'rb') as f:
        freq = pickle.load(f)
    return freq


def main(flag):
    freq = load_freq('./bert_for_qa/' + flag + 'q_tf_idf.data')
    with open('/home/ray/project/cmrc2018/cmrc2018_' + flag + '.json', 'r') as f:
        dataset = json.load(f)

    freq['第几代'] = 7
    freq['几座'] = 7
    freq['哪几'] = 7
    freq['第几个'] = 7
    freq['哪一天'] = 7
    freq['时候'] = 7

    result = []
    for entry in dataset['data']:
        paragraphs = entry['paragraphs']
        for para in paragraphs:
            qas = para['qas']
            for qa in qas:
                question = qa['question']
                word_question = list(jieba.cut(question))

                for word in word_question.copy():
                    if freq[word] < 7:
                        word_question.remove(word)

                # for word in word_question:
                #     if word in key_list:
                #         key_words += word

                result.append({
                    'question': question,
                    'key_word': ''.join(word_question)
                })
    return result


def jieba_tf(flag):
    with open('/home/ray/project/cmrc2018/cmrc2018_' + flag + '.json', 'r') as f:
        dataset = json.load(f)

    questions = ''
    word_list = []
    for entry in dataset['data']:
        paragraphs = entry['paragraphs']
        for para in paragraphs:
            context = para['context']
            qas = para['qas']
            for qa in qas:
                question = qa['question']
                questions += question
    keywords = jieba.analyse.extract_tags(questions, topK=120)
    return keywords


def save_doc_freq(flag):
    with open('/home/ray/project/cmrc2018/cmrc2018_' + flag + '.json', 'r') as f:
        dataset = json.load(f)

    word_context_list = []
    for entry in dataset['data']:
        paragraphs = entry['paragraphs']
        for para in paragraphs:
            context = para['context']
            words = list(jieba.cut(context))
            word_context_list.extend(words)
    doc_freq = Counter(word_context_list)
    with open('./bert_for_qa/WordCounter_' + flag + '.data', 'wb') as f:
        pickle.dump(doc_freq, f)
    return doc_freq


def tf_idf(flag):
    final_dict = {}
    freq = load_freq('./bert_for_qa/Counter_' + flag + '.data')
    doc_freq = load_freq('./bert_for_qa/WordCounter_' + flag + '.data')
    for word, value in freq.items():
        if word in doc_freq.keys():
            final_value = value / doc_freq[word] + 1
        else:
            final_value = value / 1
        final_dict[word] = final_value
    result = sorted(final_dict.items(), key=lambda item: item[1], reverse=True)
    result_dict = {}
    for entry in result:
        result_dict[entry[0]] = entry[1]
    with open('./bert_for_qa/' + flag + 'q_tf_idf.data', 'wb') as f:
        pickle.dump(result_dict, f)
    return result_dict


if __name__ == '__main__':
    flag = 'dev'
    save_freq(flag)
    save_doc_freq(flag)
    print(tf_idf(flag))
