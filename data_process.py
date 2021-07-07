import json

end_puc = ['?', '？', '!', '！', '.', '。']


def main(flag: str):
    with open('/home/ray/project/unilm/unilm-v1/src/qg_data/cmrc/cmrc2018_' + flag + '.json', 'r') as f:
        dataset = json.load(f)
    json_list = []
    for entry in dataset['data']:
        paragraphs = entry['paragraphs']
        for para in paragraphs:
            context = para['context']
            qas = para['qas']
            answer_dict = {}
            context_list = list(context)
            for qa in qas:
                answers = qa['answers']
                answer = answers[0]
                answer_text = answer['text']
                answer_start = answer['answer_start']

                # 每个answer在原文的位置都用[SEP]包裹起来
                try:
                    temp = answer_dict[answer_start]
                except KeyError:
                    context_list.insert(answer_start, '[SEP]')
                    context_list.insert(answer_start + len(answer_text) + 1, '[SEP]')
                else:
                    if answer_dict[answer_start] != temp:
                        print(f'something wrong! {answer_dict[answer_start]} != {temp}')
                    answer_dict[answer_start] = answer_text
                    continue
            # 把context_list标注
            context_list, label_list = label_tagging(context_list)

            last_start = 0
            for index, token in enumerate(context_list):
                if token in end_puc:
                    sentence_list = context_list[last_start:index + 1]
                    sentence_label_list = label_list[last_start:index + 1]
                    last_start = index + 1
                    if check_label_list(sentence_label_list):
                        json_list.append({
                            'tokens': sentence_list,
                            'labels': sentence_label_list})

    with open('./dataset/cmrc/cmrc_tc_sen_' + flag + '.json', 'w') as f:
        for item in json_list:
            json_item = json.dumps(item, ensure_ascii=False)
            f.write(json_item)
            f.write('\n')


def label_tagging(context_list):
    answer_flag = False
    label_list = []
    final_token_list = []
    first_flag = False
    for index, token in enumerate(context_list):
        if token != '[SEP]' and not answer_flag:
            label_list.append('O')
            final_token_list.append(token)
        elif token == '[SEP]' and not answer_flag:
            answer_flag = True
            first_flag = True
        elif token != '[SEP]' and answer_flag and first_flag:
            label_list.append('B-MISC')
            final_token_list.append(token)
            first_flag = False
        elif token != '[SEP]' and answer_flag and not first_flag:
            label_list.append('I-MISC')
            final_token_list.append(token)
        elif token == '[SEP]' and answer_flag:
            answer_flag = False
            first_flag = False
    assert len(final_token_list) == len(label_list)
    return final_token_list, label_list


def check_label_list(label_list: []):
    flag = False
    for label in label_list:
        if label == 'B-MISC':
            flag = True
    return flag


if __name__ == '__main__':
    main(flag='dev')
