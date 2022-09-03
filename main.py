import pandas as pd
import json

from natasha import (
    Segmenter,
    MorphVocab,

    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,

    PER,
    LOC,
    NamesExtractor,
    DatesExtractor,
    AddrExtractor,

    Doc
)


# зовём Наташу
segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

names_extractor = NamesExtractor(morph_vocab)
dates_extractor = DatesExtractor(morph_vocab)
addr_extractor = AddrExtractor(morph_vocab)


def ishello(txt: str) -> bool:
    """
        Проверяет содержит ли строка приветсвие менеджера
    """

    doc = Doc(txt)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    doc.tag_ner(ner_tagger)

    hello = set(["добрый",
                 "здравствовать",
                 "привет",
                 "приветствовать",
                 "почтение",
                 "рад",
                 "рада"])

    s = set()

    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        s.add(token.lemma)

    return not s.isdisjoint(hello)


def isbyebye(txt: str) -> bool:
    """
        Проверяет содержит ли строка прощание менеджера
    """

    doc = Doc(txt)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    doc.tag_ner(ner_tagger)

    bye = set(["встреча",
               "свидание",
               "хороший",
               "спасибо",
               "благодарить",
               "добрый"
               "завтра",
               "прощаться"])
    s = set()

    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        s.add(token.lemma)

    return not s.isdisjoint(bye)


def isperson(s: str) -> bool:
    """
    Проверяет есть ли слово в базе русских имён
    """

    with open("russian_names.json") as f:
        rus_names = json.load(f)
        if s.lower() in rus_names:
            return True
        else:
            return False


def get_manager_name(txt: str) -> str:
    """
        Извлекает имя менеджера, заодно проверяет представился ли он
        Если не представился, возвращает пустую строку
    """

    name = ""

    doc = Doc(txt)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    doc.tag_ner(ner_tagger)

    # try one
    # после слов "я", "это", "моё имя" обычно идёт имя человека
    for i in range(len(doc.tokens) - 1):
        if doc.tokens[i].text.lower() in ["я", "это", "имя", "меня"] and isperson(doc.tokens[i + 1].text):
            name = doc.tokens[i + 1].text

    if name: return name

    # try two
    # слева или справа от слова "зовут" может оказаться имя человека
    for i in range(1, len(doc.tokens) - 1):
        if doc.tokens[i].text == "зовут":
            lw = doc.tokens[i - 1].text
            rw = doc.tokens[i + 1].text

            if isperson(lw):
                name = lw
                break

            if isperson(rw):
                name = rw
                break

    return name


def get_company_name(txt: str) -> str:
    """
        Извлекает название компании
        Если не получилось, возвращает пустую строку
    """

    name = []

    doc = Doc(txt)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    doc.tag_ner(ner_tagger)

    i = 0
    while i < len(doc.tokens):
        doc.tokens[i].lemmatize(morph_vocab)
        if doc.tokens[i].lemma == "компания":
            break

        i += 1

    i += 1
    while i < len(doc.tokens):
        rel = doc.tokens[i].rel
        if rel in ["nsubj", "amod", "nmod"]:
            name.append(doc.tokens[i].text)
        else:
            break

        i += 1

    return " ".join(name)


def main():
    data = pd.read_csv('test_data.csv')
    data = data.groupby(['dlg_id', 'line_n']).sum()

    # извлекаем номер последнего диалога, добавляем единицу, получаем количество диалогов
    n = data.tail(1).index[0][0] + 1

    result = pd.DataFrame(columns=["Dlg_no",
                                   "Greeting",
                                   "Introduction",
                                   "Manager's name",
                                   "Company's name",
                                   "Say goodbye",
                                   "The requirement"])

    for t in range(n):
        # все реплики менеджера пакуем в список
        speech_m = data[data.role == "manager"].loc[t]
        speech_m = [x for x in speech_m.text]

        greet, intro, m_name, c_name, outro, req = None, None, None, None, None, None

        # ищем приветствие и представление себя и компании и в первых пяти репликах менеджера
        for i in range(5):
            if ishello(speech_m[i]):
                greet = speech_m[i]
            if get_manager_name(speech_m[i]):
                m_name = get_manager_name(speech_m[i])
                intro = speech_m[i]
            if get_company_name(speech_m[i]):
                c_name = get_company_name(speech_m[i])

        # ищем прощание в последних пяти репликах менеджера
        for i in range(-6, 0):
            if isbyebye(speech_m[i]):
                outro = speech_m[i]

        if greet is not None and outro is not None:
            req = True
        else:
            req = False

        result.loc[t] = [t, greet, intro, m_name, c_name, outro, req]

    result.to_csv("result_parsing.csv", index=False)


if __name__ == '__main__':
    main()

