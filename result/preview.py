import json


def extract_answers():
    with open('./result/prediction.json', 'r') as f:
        prediction_list = json.load(f)
    answer_flag = False
    start_index = 0
    answer_list = []
    for prediction in prediction_list:
        answers = []
        for index, token in enumerate(prediction):
            if token == 'B-MISC' and not answer_flag:
                if index == len(prediction) - 1:
                    answer_flag = False
                    answers.append((index, index + 1))
                else:
                    start_index = index
                    answer_flag = True

            elif token == 'B-MISC' and answer_flag:
                # 当B是最后一个token时，要把前面的上传了
                if index == len(prediction) - 1:
                    answer_flag = False
                    answers.append((index, index + 1))
                else:
                    answers.append((start_index, index))
                    start_index = index

            elif token == 'O' and answer_flag:
                answers.append((start_index, index))
                answer_flag = False
            elif token == 'I-MISC' and answer_flag and index == len(prediction) - 1:
                answers.append((start_index, index + 1))
                answer_flag = False
        answer_list.append(answers)
    with open('./result/answers.json', 'w') as f:
        json.dump(answer_list, f, indent=4)


def back2word():
    with open('./result/answers.json', 'r') as f:
        answer_list = json.load(f)

    pred_list = []
    with open('./dataset/cmrc/cmrc_tc_sen_dev.json', 'r') as f:
        for line in f:
            item = json.loads(line)
            pred_list.append(item)
    result_list = []
    for index, answers in enumerate(answer_list):
        answers_text = []
        for answer in answers:
            answer_text = pred_list[index]['tokens'][answer[0]: answer[1]]
            answer_text = ''.join(answer_text)
            answers_text.append(answer_text)
        result_list.append({
            'context': ''.join(pred_list[index]['tokens']),
            'pred_answers': answers_text
        })
    with open('./result/result.json', 'w') as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    back2word()
