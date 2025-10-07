import pytest
from vectordb.splitters.sentence_splitter import SentenceSpliter


@pytest.fixture(scope='module')
def splitter():
    return SentenceSpliter()


def test_quotes(splitter):
    text = 'Он сказал: "Привет!" И ушёл.'
    sentences = splitter.split(text)
    assert sentences == ['Он сказал: "Привет!"', "И ушёл."]

def test_question_and_exclamation(splitter):
    text = "Что происходит?! Не понимаю!"
    sentences = splitter.split(text)
    assert sentences == ["Что происходит?!", "Не понимаю!"]

def test_numbers_with_dots(splitter):
    text = "Значение π ≈ 3.14. Это важно."
    sentences = splitter.split(text)
    assert sentences == ["Значение π ≈ 3.14.", "Это важно."]

def test_multiple_abbreviations(splitter):
    text = "Встретимся на ул. Ленина, д. 10. Затем пойдем в парк."
    sentences = splitter.split(text)
    assert sentences == ["Встретимся на ул. Ленина, д. 10.", "Затем пойдем в парк."]

def test_parentheses(splitter):
    text = "Он приехал (в Москву). И сразу пошёл на встречу."
    sentences = splitter.split(text)
    assert sentences == ["Он приехал (в Москву).", "И сразу пошёл на встречу."]

def test_mixed_language(splitter):
    text = "This is English. А это русский."
    sentences = splitter.split(text)
    assert sentences == ["This is English.", "А это русский."]