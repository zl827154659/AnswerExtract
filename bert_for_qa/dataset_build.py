import json

import jieba

from bert_for_qa.word_count import load_freq

end_puc = ['?', '？', '!', '！', '.', '。']


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

    freq['出生地'] = 7
    freq['多少倍'] = 7
    freq['哪几个'] = 7
    freq['哪一'] = 7
    freq['特点'] = 7
    freq['如何'] = 7
    freq['哪几种'] = 7
    freq['几个'] = 7
    freq['名字'] = 7
    freq['哪支'] = 7
    freq['哪年'] = 7
    freq['几条'] = 7
    freq['几届'] = 7
    freq['哪一科'] = 7

    result = []
    for index, entry in enumerate(dataset['data']):
        paragraphs = entry['paragraphs']
        for para in paragraphs:
            context = para['context']
            qas = para['qas']
            for qa in qas:
                question = qa['question']
                id = qa['id']

                word_question = list(jieba.cut(question))

                for word in word_question.copy():
                    if freq[word] < 7:
                        word_question.remove(word)

                answer = qa['answers'][0]
                answer_text = answer['text']
                answer_start = answer['answer_start']
                answer_end = answer_start + len(answer_text)
                context_start = 0
                context_end = len(context)
                for i in range(answer_start, 0, -1):
                    if context[i] in end_puc:
                        context_start = i + 1
                        break
                for j in range(answer_end - 1, context_end, 1):
                    if context[j] in end_puc:
                        context_end = j + 1
                        break
                new_context = context[context_start:context_end]

                result.append({
                    'context': new_context,
                    'question': ''.join(word_question),
                    'answer_text': answer_text,
                    'answer_start': new_context.index(answer_text),
                    'id': id
                })
    with open('./bert_for_qa/dataset/qa_' + flag + '.json', 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main('dev')
