#  -*- coding: utf-8 -*-

import json
from pathlib import Path


def load_label_map(map_dir="./data/label_map.json"):
    """
    :param map_dir: dict indictuing chunk type
    :return:
    """
    return json.load(open(map_dir, "r"))

def json_to_text(file_path,data):
    '''
    将json list写入text文件中
    :param file_path:
    :param data:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'w') as fw:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            fw.write(line + '\n')

def cal_chunk(pred_label, refer_label):
    tp = dict()
    fn = dict()
    fp = dict()
    for i in range(len(refer_label)):
        if refer_label[i] == pred_label[i]:
            if refer_label[i] not in tp:
                tp[refer_label[i]] = 0
            tp[refer_label[i]] += 1
        else:
            if pred_label[i] not in fp:
                fp[pred_label[i]] = 0
            fp[pred_label[i]] += 1
            if refer_label[i] not in fn:
                fn[refer_label[i]] = 0
            fn[refer_label[i]] += 1

    tp_total = sum(tp.values())
    fn_total = sum(fn.values())
    fp_total = sum(fp.values())
    p_total = float(tp_total) / (tp_total + fp_total)
    r_total = float(tp_total) / (tp_total + fn_total)
    f_micro = 2 * p_total * r_total / (p_total + r_total)

    return f_micro
def get_entity_bios(seq,id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def res_evaluate(res_dir="./outputs/predict/predictions.json", data_dir="./data/test.tsv"):
    label_map = load_label_map()
    num_map = {}
    num_map[0] = 'B-trigger'
    num_map[1] = 'I-trigger'
    num_map[2] = 'S-trigger'
    num_map[3] = 'O'
    id2label = ['B-trigger', 'I-trigger', 'S-trigger', 'O']
    print(label_map)
    print(num_map)

    total_label = []
    total_text = []
    with open(data_dir, "r") as file:
        first_flag = True
        for line in file:
            if first_flag:
                first_flag = False
                continue
            line = line.strip("\n")
            if len(line) == 0:
                continue
            line = line.split("\t")
            if len(line) < 2:
                continue
            labels = line[1].split("\x02")
            text = line[0].split("\x02")
            total_label.append(labels)
            total_text.append(text)
    total_label = [[label_map[j] for j in i] for i in total_label]
    print(total_text[0])
  
    total_res = []
    total_label_entity = []
    with open(res_dir, "r") as file:
        cnt = 0
        for line in file:
            line = line.strip("\n")
            if len(line) == 0:
                continue
            try:
                res_arr = json.loads(line)

                if len(total_label[cnt]) < len(res_arr):
                    total_res.append(res_arr[1: 1 + len(total_label[cnt])])
                    res_arr = res_arr[1: 1 + len(total_label[cnt])]
                elif len(total_label[cnt]) == len(res_arr):
                    total_res.append(res_arr)
                else:
                    total_res.append(res_arr)
                    total_label[cnt] = total_label[cnt][: len(res_arr)]
            except:
                print("json format error: {}".format(cnt))
                print(line)
            label_entity = get_entity_bios(res_arr,id2label)
            total_label_entity.append(label_entity)

            cnt += 1
    total_res_label = [[num_map[j] for j in i] for i in total_res]
    print(total_label_entity[0])

    total_res_equal = []
    total_label_equal = []
    assert len(total_label) == len(total_res), "prediction result doesn't match to labels"
    for i in range(len(total_label)):
        num = len(total_label[i])
        total_label_equal.extend(total_label[i])
        total_res[i] = total_res[i][:num]
        total_res_equal.extend(total_res[i])

    f1 = cal_chunk(total_res_equal, total_label_equal)
    print('data num: {}'.format(len(total_label)))
    print("f1: {:.4f}".format(f1))
    print(total_label[0])
    print(total_res[0])
   
    test_submit = [] 
    output_submit_file = './outputs/trigger_result.json'
    for x, y in zip(total_text, total_label_entity):
        json_d = {}
        json_d['text'] = ''.join(x) 
        json_d['label'] = {}
        entities = y
        words = x
        if len(entities) != 0:
            for subject in entities:
                tag = subject[0]
                start = subject[1]
                end = subject[2]
                word = "".join(words[start:end + 1])
                if tag in json_d['label']:
                    if word in json_d['label'][tag]:
                        json_d['label'][tag][word].append([start, end])
                    else:
                        json_d['label'][tag][word] = [[start, end]]
                else:
                    json_d['label'][tag] = {}
                    json_d['label'][tag][word] = [[start, end]]
        test_submit.append(json_d)
    json_to_text(output_submit_file,test_submit)

res_evaluate()
