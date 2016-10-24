#coding:utf-8
import re
import jieba
from stp_nlp import stp_seg
from hanziconv import HanziConv

# 文本长度小于指定长度的进行丢弃
def text_filter(corpus, length=140):
    return [text for text in corpus if len(text)>length]

# 繁简转换:把繁体转换为简体
def traditional_to_simplified(ustring):
    return HanziConv.toSimplified(ustring)

# 全角转半角(中文文字永远是全角，只有英文字母、数字键、符号键才有全角半角的概念
# 一个字母或数字占一个汉字的位置叫全角，占半个汉字的位置叫半角。)
def quan_to_ban(ustring):
    banjiao = ''
    for uchar in ustring:
        int_ordinal = ord(uchar)  #the integer ordinal of a one-character string
        if int_ordinal == 12288:
            int_ordinal = 32
        elif (int_ordinal >= 65281) and (int_ordinal <= 65374):
            int_ordinal -= 65248
        banjiao += unichr(int_ordinal)
    return banjiao

# 文本分句
def sentence_cut(text):
    sentences = []
    sentiment_word_position = 0
    word_position = 0
    punctuation_list = ',.!?;~，。！？；～… '.decode('utf8')
    for words in text:
        word_position += 1
        if words in punctuation_list:
            nextWord = list(text[sentiment_word_position:word_position+1]).pop()
            if nextWord not in punctuation_list:
                sentences.append(text[sentiment_word_position:word_position])
                sentiment_word_position = word_position
    if sentiment_word_position < len(text):
        sentences.append(text[sentiment_word_position:])
    return sentences

# 删除数字/将数字统一转换为1
def number_processing(ustring, str=''):
    if str == '':
        return re.sub('\d+', '', ustring)
    else:
        return re.sub('\d+', '1', ustring)

# 去除特殊字符
def char_filter(ustring):
    special_char = u"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）)《》-『』<>]"
    return [term for term in ustring if term not in special_char]

# 去除标点符号
def punctuate_filter(ustring):
    punctuation_list = u',.!?;~，。！？；:～… “”‘’'
    return [term for term in ustring if term not in punctuation_list]

# 分词
def word_segment(utext, param='stp'):
    if param == 'stp':
        text = utext.encode('utf-8')
        return ' '.join([term.decode('utf-8') for term in stp_seg.cut(text)])
    elif param == 'jieba':
        return ' '.join(jieba.cut(utext))

def word_segmentation(utext, param='jieba'):
    if param == 'stp':
        text = utext.encode('utf-8')
        return [term.decode('utf-8') for term in stp_seg.cut(text)]
    elif param == 'jieba':
        return list(jieba.cut(utext))

def whole_process(filepath):
    text = []
    for line in file(filepath):
        ustring = traditional_to_simplified(line.strip())
        ustring = quan_to_ban(ustring)
        ustring = number_processing(ustring)
        ustring = word_segmentation(ustring)
        ustring = punctuate_filter(ustring)
        ustring = char_filter(ustring)
        text.append(' '.join(ustring))
    return text


